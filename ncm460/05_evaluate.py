# evaluate the trained NCM460 models on the validation and test sequences.
import pandas as pd
import torch

import config as cfg
import data_loading
import evaluation
import models
import run_utils


def fmt_ci(estimate, ci, digits=2):
    return f"{estimate:+.{digits}f} [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def gain_row(label, out, new_name="full_gate", base_name="appearance"):
    # Row of the gain table: fused model against Stage 1 alone, with bootstrap interval and McNemar test.
    preds = evaluation.prediction_variants(out)
    base, new = preds[base_name], preds[new_name]
    boot = evaluation.well_bootstrap_gain(out["labels"], base, new, out["well"])
    fixed, broken, p = evaluation.exact_mcnemar(out["labels"], base, new)
    top1_base, f1_base = run_utils.top1_and_macro_f1(out["labels"], base)
    top1_new, f1_new = run_utils.top1_and_macro_f1(out["labels"], new)
    return {"model": label, "n": len(out["labels"]), "wells": boot["n_wells"], "baseline top-1 (%)": round(top1_base, 2),
            "fused top-1 (%)": round(top1_new, 2), "gain top-1 (pp) [95% CI]": fmt_ci(boot["top1_gain"], boot["top1_ci"]),
            "baseline macro-F1": round(f1_base, 4), "fused macro-F1": round(f1_new, 4),
            "gain macro-F1 [95% CI]": fmt_ci(boot["f1_gain"], boot["f1_ci"], 4), "corrected": fixed, "broken": broken, "McNemar p": f"{p:.2e}"}


def as_markdown(df, index=False):
    try:
        return df.to_markdown(index=index)
    except Exception:
        return "```\n" + df.to_string(index=index) + "\n```"


def main():
    device = models.device()
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {seed: run_utils.registered_run("stage3", seed) for seed in cfg.SEEDS}
    missing = [seed for seed, run in runs.items() if run is None]
    if missing:
        raise RuntimeError(f"no registered Stage 3 run for the seeds {missing}; run 04_train_stage3_gate.py first")

    outputs = {split: {} for split in cfg.EVAL_SPLITS}
    for seed, run in runs.items():
        model = models.load_stage3_checkpoint(run / "best_model.pth")
        for split in cfg.EVAL_SPLITS:
            records = data_loading.load_index(cfg.NCM_INDEX, split, min_T=cfg.MIN_T)
            print(f"\nseed {seed}, {split}: {len(records)} sequences")
            # the experts are identical in every seed: their outputs are read from runs/stage3_features (or computed once), only the gate differs
            out = evaluation.fused_outputs(model, evaluation.cached_features(split, records, model, device), device)
            evaluation.save_arrays(out, run / f"{split}_features.npz")
            table = evaluation.summary_table(out)
            table.to_csv(run / f"{split}_summary.csv", index=False)
            print(table.round(4).to_string(index=False))
            outputs[split][seed] = out
        del model
        torch.cuda.empty_cache()

    # ---- test split: per-seed gains, the combined model, per-class recall ----
    test = outputs["test"] if "test" in outputs else {}
    if not test:
        print("The test split is not in EVAL_SPLITS; no report written.")
        return
    ensemble = evaluation.ensemble_outputs([test[s] for s in cfg.SEEDS])
    evaluation.save_arrays(ensemble, cfg.REPORT_DIR / "ncm460_test_features_mean_of_seeds.npz")

    gains = pd.DataFrame([gain_row(f"seed {s}", test[s]) for s in cfg.SEEDS] + [gain_row(f"mean of {len(cfg.SEEDS)} gates", ensemble)])
    # the fused model against the graph expert alone, and the graph expert against Stage 1 (is the fusion more than its stronger expert?)
    comparisons = pd.DataFrame([gain_row("fused vs graph alone", ensemble, "full_gate", "graph"),
                                gain_row("graph alone vs Stage 1", ensemble, "graph", "appearance")])
    variants = evaluation.summary_table(ensemble).round({"top1": 2, "macro_f1": 4})
    recall = evaluation.per_class_recall(ensemble, {k: v for k, v in evaluation.prediction_variants(ensemble).items()
                                                  if k in ("appearance", "graph", "full_gate")}).round(1)
    recall.to_csv(cfg.REPORT_DIR / "ncm460_per_class_recall.csv")

    text = ["# NCM460: train, validation and test on NCM460 (10 classes)", "",
            f"Test sequences with at least {cfg.MIN_T} frames: {len(ensemble['labels'])} from {gains['wells'].iloc[-1]} wells. "
            f"Intervals: 95% percentile, whole-well bootstrap, {cfg.N_BOOT} replicates, seed {cfg.BOOT_SEED}.", "",
            "## Gain of the fused model over Stage 1 alone", "", as_markdown(gains), "",
            "## Fused model against the graph expert alone, and graph expert against Stage 1 (mean of the gates)", "", as_markdown(comparisons), "",
            "## All variants (mean of the three gates)", "", as_markdown(variants), "",
            "## Recall per class (%)", "", as_markdown(recall, index=True), ""]
    (cfg.REPORT_DIR / "ncm460_results.md").write_text("\n".join(text), encoding="utf-8")
    print("\n" + "\n".join(text))
    print(f"written: {cfg.REPORT_DIR / 'ncm460_results.md'}")


if __name__ == "__main__":
    main()
