from functools import partial
from jax import random
import jax.numpy as np
from jax.scipy.linalg import block_diag
from flax import serialization
import os, wandb
import numpy as _np  # <-- real NumPy for disk I/O


from .train_helpers import create_train_state, reduce_lr_on_plateau,\
    linear_warmup, cosine_annealing, constant_lr, train_epoch, validate
from .dataloading import Datasets
from .seq_model import BatchClassificationModel, RetrievalModel
from .ssm import init_S5SSM
from .ssm_init import make_DPLR_HiPPO


# -----------------------------
# SIMPLE LOCAL CHECKPOINT CONFIG
# -----------------------------
CKPT_DIR = "./checkpoints"
BEST_CKPT_PATH = os.path.join(CKPT_DIR, "best_state.msgpack")

# Change just this one flag as you wish:
#   "TRAIN"      -> resume from BEST_CKPT_PATH if present, else fresh train, and keep rewriting best
#   "PRUNE_ONLY" -> load BEST_CKPT_PATH and run pruning immediately (no training)
MODE = "TRAIN"   # or "PRUNE_ONLY" "TRAIN"


import csv
import json

def _topk_mask_desc(x, k):
    """Return boolean mask for the top-k elements of x (desc), tie-broken by index."""
    x_np = _np.asarray(x).ravel()
    N = x_np.size
    # Sort by (-score, -index) so earlier indices win ties deterministically
    order = _np.lexsort((_np.arange(N)[::-1], -x_np))
    topk_idx = order[:max(k,1)]
    mask = _np.zeros(N, dtype=bool)
    mask[topk_idx] = True
    return mask, topk_idx


def _effective_states_per_layer(args):
    """
    Effective P (states per layer) used to build layer slices for ENERGYscores.
    Matches the 'total_states' logic in your code.
    """
    return int(args.ssm_size_base * (0.5 if args.conj_sym else 1.0))

def _layer_slices(n_layers, states_per_layer):
    """Yield (l, slice) for each layer's chunk inside the flat ENERGYscores vector."""
    for l in range(n_layers):
        start = l * states_per_layer
        end = start + states_per_layer
        yield l, slice(start, end)

def _layerwise_stats(args, AIRE_scores_flat, global_th):
    """
    Compute per-layer keep/prune counts, prune %, kept-energy %, and the
    smallest kept score in the layer (layer_threshold).
    Returns: (stats_list, global_mask)
        stats_list: [ {layer, keep, pruned, prune_pct, energy_kept_pct, layer_threshold}, ... ]
        global_mask: boolean mask over AIRE_scores_flat for states kept globally
    """
    P_eff = _effective_states_per_layer(args)
    L = int(args.n_layers)
    N = AIRE_scores_flat.shape[0]
    assert N == P_eff * L, f"ENERGYscores length {N} != P_eff*L {P_eff}*{L}"

    # Keep states with score >= global_th (same semantics as your thresholding)
    keep_mask = (AIRE_scores_flat >= global_th)

    stats = []
    for l, sl in _layer_slices(L, P_eff):
        scores_l = AIRE_scores_flat[sl]
        mask_l   = keep_mask[sl]
        keep     = int(mask_l.sum())
        total    = int(scores_l.size)
        pruned   = total - keep
        prune_pct = (pruned / total) * 100.0

        # Energy fractions computed on AIRE scores (already normalized per AIRE definition)
        energy_total = float(scores_l.sum()) + 1e-12
        energy_kept  = float(scores_l[mask_l].sum())
        energy_kept_pct = (energy_kept / energy_total) * 100.0

        # Smallest score among kept in this layer (NaN if nothing kept)
        layer_th = float(scores_l[mask_l].min()) if keep > 0 else float('nan')

        stats.append({
            "layer": int(l),
            "keep": keep,
            "pruned": pruned,
            "prune_pct": prune_pct,
            "energy_kept_pct": energy_kept_pct,
            "layer_threshold": layer_th,
        })
    return stats, keep_mask

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _save_best_state(state, path=BEST_CKPT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))
    print(f"[*] Saved BEST checkpoint → {path}")


def _try_restore_best_state(state, path=BEST_CKPT_PATH):
    try:
        with open(path, "rb") as f:
            data = f.read()
        state = serialization.from_bytes(state, data)
        print(f"[*] Restored state from {path}")
        return state, True
    except FileNotFoundError:
        print(f"[*] No checkpoint at {path} (fresh run).")
        return state, False
    except Exception as e:
        print(f"[!] Failed to restore from {path}: {e}")
        return state, False


def _run_pruning_sweep(args, state, model_cls, testloader, seq_len, in_dim, ENERGYscores):
    PRUNING_SWEEP = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    if not args.pruning:
        print("[*] Pruning disabled (args.pruning=False). Skipping sweep.")
        return

    # --- Prepare scores and sizes ---
    AIRE_flat = ENERGYscores[0]  # flat vector (total_states,)
    P_eff = _effective_states_per_layer(args)
    L = int(args.n_layers)
    total_states = P_eff * L
    assert int(AIRE_flat.shape[0]) == int(total_states), \
        f"Expected {total_states} AIRE scores, got {AIRE_flat.shape[0]}"

    # For threshold reporting only
    sorted_scores = np.sort(AIRE_flat)  # JAX ok here

    # --- Output dirs & CSV ---
    os.makedirs(CKPT_DIR, exist_ok=True)
    masks_dir = os.path.join(CKPT_DIR, "prune_masks"); os.makedirs(masks_dir, exist_ok=True)
    csv_path  = os.path.join(CKPT_DIR, "prune_sweep_layerwise_listops.csv")
    first_csv_write = not os.path.exists(csv_path)

    # W&B table to collect per-ratio, per-layer summaries
    table_cols = ["ratio_pct", "kth_score"]
    table_cols += [f"layer{l}_prune_pct" for l in range(L)]
    table_cols += [f"layer{l}_keep" for l in range(L)]
    table_cols += [f"layer{l}_energy_kept_pct" for l in range(L)]
    layerwise_table = wandb.Table(columns=table_cols)

    sweep_results = []
    print("\n=>> Pruning sweep (global) ===")
    for ratio in PRUNING_SWEEP:
        keep = max(int(total_states * (100.0 - float(ratio)) / 100.0), 1)

        # ---------- USE TOP-K (deterministic) ----------
        keep_mask_np, topk_idx = _topk_mask_desc(AIRE_flat, keep)
        actual_kept = int(keep_mask_np.sum())
        # kth score for display (not used to build the mask)
        global_th = float(_np.asarray(AIRE_flat)[topk_idx].min())

        print(f"\n  -> Ratio: {ratio:.1f}% | Target keep: {keep} | Actual keep (TopK): {actual_kept} "
              f"| kth score: {global_th:.12g}")
        print("     Evaluating pruning...")

        # You can keep your validate() call driven by a threshold; harmless for accuracy
        # For consistency we still compute a threshold from sorted_scores:
        # (not used for masks/stats; only for eval path if your validate() needs it)
        eval_th = float(sorted_scores[-keep]) if keep < total_states else float(sorted_scores[0])
        test_loss_i, test_acc_i, _ = validate(
            state, model_cls, testloader, seq_len, in_dim, args.batchnorm, global_th=eval_th
        )
        print(f"     Test Loss: {test_loss_i:.5f} | Test Acc: {test_acc_i:.4f}")

        # ---------- Layer-wise stats from the Top-K mask ----------
        layer_stats = []
        for l, sl in _layer_slices(L, P_eff):
            scores_l = AIRE_flat[sl]
            mask_l   = keep_mask_np[sl]
            keep_l   = int(mask_l.sum())
            total_l  = int(scores_l.size)
            pruned_l = total_l - keep_l
            prune_pct = (pruned_l / total_l) * 100.0

            energy_total = float(_np.asarray(scores_l).sum())
            if energy_total == 0.0:
                energy_kept_pct = 0.0
                if ratio == PRUNING_SWEEP[0]:
                    print(f"[warn] Layer {l} has zero AIRE energy; scores likely all zeros.")
            else:
                energy_kept  = float(_np.asarray(scores_l)[_np.asarray(mask_l, dtype=bool)].sum())
                energy_kept_pct = (energy_kept / energy_total) * 100.0

            layer_th = float(_np.asarray(scores_l)[_np.asarray(mask_l, dtype=bool)].min()) if keep_l > 0 else float('nan')

            layer_stats.append({
                "layer": l,
                "keep": keep_l,
                "pruned": pruned_l,
                "prune_pct": prune_pct,
                "energy_kept_pct": energy_kept_pct,
                "layer_threshold": layer_th,
            })

        # Pretty print table
        print("     ── Layer-wise stats (Top-K mask applied) ────────────────────────────────────")
        print("     layer | keep | pruned | prune%  | energy_kept% | layer_threshold")
        for s in layer_stats:
            print(f"     {s['layer']:>5} | {s['keep']:>4} | {s['pruned']:>6} | "
                  f"{s['prune_pct']:>6.2f} | {s['energy_kept_pct']:>12.2f} | {s['layer_threshold']:.6g}")
        print("     ────────────────────────────────────────────────────────────────────────────")

        # W&B scalars
        log_dict = {
            "prune/ratio_pct": float(ratio),
            "prune/target_keep_states": int(keep),
            "prune/actual_keep_states": int(actual_kept),
            "prune/kth_score": float(global_th),
            "prune/test_loss": float(test_loss_i),
            "prune/test_acc": float(test_acc_i),
        }
        for s in layer_stats:
            l = s["layer"]
            log_dict.update({
                f"prune/layer_{l}/prune_pct": s["prune_pct"],
                f"prune/layer_{l}/keep": s["keep"],
                f"prune/layer_{l}/energy_kept_pct": s["energy_kept_pct"],
                f"prune/layer_{l}/layer_threshold": s["layer_threshold"],
            })
        wandb.log(log_dict)

        # W&B table row
        row = [float(ratio), float(global_th)]
        row += [s["prune_pct"] for s in layer_stats]
        row += [s["keep"] for s in layer_stats]
        row += [s["energy_kept_pct"] for s in layer_stats]
        layerwise_table.add_data(*row)

        # Save masks (use real NumPy I/O)
        layer_masks = {}
        for l in range(L):
            sl = slice(l*P_eff, (l+1)*P_eff)
            layer_masks[f"layer{l}"] = _np.asarray(keep_mask_np[sl], dtype=bool)
        _np.savez_compressed(os.path.join(masks_dir, f"ratio_{int(ratio):02d}.npz"), **layer_masks)

        # Append CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if first_csv_write:
                header = ["ratio_pct", "kth_score", "layer", "keep", "pruned",
                          "prune_pct", "energy_kept_pct", "layer_threshold",
                          "target_keep_states", "actual_keep_states"]
                writer.writerow(header)
                first_csv_write = False
            for s in layer_stats:
                writer.writerow([
                    float(ratio), float(global_th), s["layer"], s["keep"], s["pruned"],
                    s["prune_pct"], s["energy_kept_pct"], s["layer_threshold"],
                    keep, actual_kept
                ])

        sweep_results.append({
            "ratio": float(ratio),
            "threshold(kth)": float(global_th),
            "test_loss": float(test_loss_i),
            "test_acc": float(test_acc_i),
            "actual_keep": int(actual_kept),
        })

    wandb.log({"prune/layerwise_table": layerwise_table})
    wandb.save(csv_path)

    if sweep_results:
        best_row = max(sweep_results, key=lambda r: r["test_acc"])
        wandb.run.summary["prune_sweep/best_ratio_pct"] = best_row["ratio"]
        wandb.run.summary["prune_sweep/best_test_acc"] = best_row["test_acc"]
        wandb.run.summary["prune_sweep/best_kth_score"] = best_row["threshold(kth)"]
        wandb.run.summary["prune_sweep/csv_path"] = csv_path

def train(args):
    """
    Training with a single-file, fixed-name checkpoint:
    - MODE == "TRAIN": resume if available, else fresh; keep overwriting BEST_CKPT_PATH on improvement.
    - MODE == "PRUNE_ONLY": load BEST_CKPT_PATH and prune immediately.
    """

    best_test_loss = 1e9
    best_test_acc  = -1e9

    if args.USE_WANDB:
        wandb.init(project=args.wandb_project, job_type='model_training',
                   config=vars(args), entity=args.wandb_entity)
    else:
        wandb.init(mode='offline')

    ssm_size = args.ssm_size_base
    ssm_lr   = args.ssm_lr_base

    block_size = int(ssm_size / args.blocks)
    wandb.log({"block_size": block_size})

    lr = args.lr_factor * ssm_lr

    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    create_dataset_fn = Datasets[args.dataset]

    if args.dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
        padded = True
        retrieval = args.dataset in ["aan-classification"]
        if retrieval:
            print("Using retrieval model for document matching")
    else:
        padded = False
        retrieval = False

    speech = args.dataset in ["speech35-classification"]
    if speech:
        print("Will evaluate on both resolutions for speech task")

    init_rng, key = random.split(init_rng, num=2)
    trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
        create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    print(f"[*] Starting S5 on `{args.dataset}` =>> Initializing...")

    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size //= 2
        ssm_size   //= 2

    Lambda = Lambda[:block_size]
    V      = V[:, :block_size]
    Vc     = V.conj().T

    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V      = block_diag(*([V] * args.blocks))
    Vinv   = block_diag(*([Vc] * args.blocks))

    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))

    ssm_init_fn = init_S5SSM(H=args.d_model,
                             P=ssm_size,
                             Lambda_re_init=Lambda.real,
                             Lambda_im_init=Lambda.imag,
                             V=V,
                             Vinv=Vinv,
                             C_init=args.C_init,
                             discretization=args.discretization,
                             dt_min=args.dt_min,
                             dt_max=args.dt_max,
                             conj_sym=args.conj_sym,
                             clip_eigs=args.clip_eigs,
                             bidirectional=args.bidirectional,
                             pruning=args.pruning)

    if retrieval:
        model_cls = partial(
            RetrievalModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )
    else:
        model_cls = partial(
            BatchClassificationModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    state = create_train_state(model_cls,
                               init_rng,
                               padded,
                               retrieval,
                               in_dim=in_dim,
                               bsz=args.bsz,
                               seq_len=seq_len,
                               weight_decay=args.weight_decay,
                               batchnorm=args.batchnorm,
                               opt_config=args.opt_config,
                               ssm_lr=ssm_lr,
                               lr=lr,
                               dt_global=args.dt_global)

    # ---------------------------
    # PRUNE ONLY: load and prune
    # ---------------------------
    if MODE.upper() == "PRUNE_ONLY":
        state, ok = _try_restore_best_state(state, BEST_CKPT_PATH)
        if not ok:
            print("[!] PRUNE_ONLY requested but no checkpoint found. Aborting prune.")
            return
        print("[*] Recomputing scores before pruning...")
        test_loss, test_acc, ENERGYscores = validate(state, model_cls, testloader, seq_len, in_dim, args.batchnorm)
        print(f"[*] Baseline (restored) Test: loss={test_loss:.5f}, acc={test_acc:.4f}")
        _run_pruning_sweep(args, state, model_cls, testloader, seq_len, in_dim, ENERGYscores)
        return

    # ---------------------------
    # TRAIN: resume if available
    # ---------------------------
    state, _ = _try_restore_best_state(state, BEST_CKPT_PATH)  # okay if missing; we'll train fresh

    best_loss, best_acc, best_epoch = 1e9, -1e9, 0
    count, best_val_loss = 0, 1e9
    lr_count, opt_acc = 0, -1e9
    step = 0
    steps_per_epoch = int(train_size / args.bsz)

    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        if epoch < args.warmup_end:
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end
            print(f"using linear warmup for epoch {epoch+1}")
        elif args.cosine_anneal:
            decay_function = cosine_annealing
            end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
            print(f"using cosine annealing for epoch {epoch+1}")
        else:
            decay_function = constant_lr
            end_step = None
            print(f"using constant lr for epoch {epoch+1}")

        lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)

        train_rng, skey = random.split(train_rng)
        state, train_loss, step = train_epoch(state, skey, model_cls, trainloader, seq_len, in_dim, args.batchnorm, lr_params)

        if valloader is not None:
            val_loss, val_acc, ENERGYscores = validate(state, model_cls, valloader, seq_len, in_dim, args.batchnorm)
            test_loss, test_acc, _       = validate(state, model_cls, testloader, seq_len, in_dim, args.batchnorm)
            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} --Test Loss: {test_loss:.5f} --"
                  f" Val Accuracy: {val_acc:.4f} Test Accuracy: {test_acc:.4f}")
        else:
            val_loss, val_acc, ENERGYscores = validate(state, model_cls, testloader, seq_len, in_dim, args.batchnorm)
            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} -- Test Accuracy: {val_acc:.4f}")

        # Early stopping tracking
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        # New best → save single fixed file
        if val_acc > best_acc:
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

            _save_best_state(state, BEST_CKPT_PATH)

            if speech:
                val2_loss, val2_acc = validate(state, model_cls, aux_dataloaders['valloader2'],
                                               int(seq_len // 2), in_dim, args.batchnorm, step_rescale=2.0)
                test2_loss, test2_acc = validate(state, model_cls, aux_dataloaders['testloader2'],
                                                 int(seq_len // 2), in_dim, args.batchnorm, step_rescale=2.0)
                print(f"\n=>> Epoch {epoch + 1} Res 2 Metrics ===")
                print(f"\tVal2 Loss: {val2_loss:.5f} --Test2 Loss: {test2_loss:.5f} --"
                      f" Val Accuracy: {val2_acc:.4f} Test Accuracy: {test2_acc:.4f}")

        # LR schedule
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(
            (lr, ssm_lr, lr_count, val_acc, opt_acc),
            factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min
        )

        print(f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy: {best_acc:.4f} at Epoch {best_epoch + 1}\n"
              f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy: {best_test_acc:.4f} at Epoch {best_epoch + 1}\n")

        # WANDB logging
        if valloader is not None:
            wandb.log({
                "Training Loss": train_loss, "Val loss": val_loss, "Val Accuracy": val_acc,
                "Test Loss": test_loss, "Test Accuracy": test_acc,
                "count": count, "Learning rate count": lr_count, "Opt acc": opt_acc,
                "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'],
            })
        else:
            wandb.log({
                "Training Loss": train_loss, "Val loss": val_loss, "Val Accuracy": val_acc,
                "count": count, "Learning rate count": lr_count, "Opt acc": opt_acc,
                "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'],
            })

        wandb.run.summary["Best Val Loss"] = best_loss
        wandb.run.summary["Best Val Accuracy"] = best_acc
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        wandb.run.summary["Best Test Accuracy"] = best_test_acc

        if count > args.early_stop_patience:
            print("[*] Early stopping triggered.")
            break

    # After training, optionally prune using the best ckpt we just kept
    if args.pruning:
        # reload best to be safe (ensures we prune the actual best weights)
        state, _ = _try_restore_best_state(state, BEST_CKPT_PATH)
        print("[*] Recomputing scores on test set before pruning...")
        test_loss, test_acc, ENERGYscores = validate(state, model_cls, testloader, seq_len, in_dim, args.batchnorm)
        print(f"[*] (best) Test: loss={test_loss:.5f}, acc={test_acc:.4f}")
        _run_pruning_sweep(args, state, model_cls, testloader, seq_len, in_dim, ENERGYscores)
