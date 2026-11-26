#!/usr/bin/env bash
# Trains and evaluates the temporal-only control experts on a public dataset, then compares them with the patch graph.
# Run from external_datasets/ after the graph runs of the same folds (04-07) have finished: their Stage 1 is reused.
#   bash run_temporal_control.sh                                   # LCTP, held-out compounds, 4 folds, control "temporal"
#   PUBLIC_PROTOCOL=plate FOLDS="0 1 2" bash run_temporal_control.sh
#   ARCHS="temporal temporal_nodes" bash run_temporal_control.sh
#   PUBLIC_DATASET=pastis PUBLIC_PROTOCOL=official5 FOLDS="0 1 2 3 4" bash run_temporal_control.sh
# Set the same data variables as for the graph runs (PUBLIC_LCTP_ROOT, PUBLIC_TILE_CACHE, PUBLIC_OUT_DIR, ...). A stage that is already registered is skipped
# (PUBLIC_RETRAIN=1 trains it again), so an interrupted run continues when the command is started again.
set -u
export PUBLIC_DATASET="${PUBLIC_DATASET:-lctp_a549}"
export PUBLIC_PROTOCOL="${PUBLIC_PROTOCOL:-compound}"
default_folds="0 1 2 3"
[ "$PUBLIC_PROTOCOL" = "plate" ] && default_folds="0 1 2"
[ "$PUBLIC_PROTOCOL" = "official5" ] && default_folds="0 1 2 3 4"
FOLDS="${FOLDS:-$default_folds}"
ARCHS="${ARCHS:-temporal}"
OUT_DIR="${PUBLIC_OUT_DIR:-.}"

# the graph arm of every fold must exist: the controls reuse its Stage 1 and the comparison reads its test outputs
for fold in $FOLDS; do
  graph_run="$OUT_DIR/runs/${PUBLIC_DATASET}_${PUBLIC_PROTOCOL}_f${fold}"
  grep -q '"stage1"' "$graph_run/registry.json" 2>/dev/null || { echo "missing Stage 1 of the graph run of fold $fold ($graph_run/registry.json)"; exit 1; }
  [ -f "$OUT_DIR/reports/${PUBLIC_DATASET}_${PUBLIC_PROTOCOL}_f${fold}/test_outputs.npz" ] || { echo "missing test outputs of the graph run of fold $fold (07_evaluate.py)"; exit 1; }
done

for arch in $ARCHS; do
  for fold in $FOLDS; do
    echo "=== $arch, fold $fold ==="
    export PUBLIC_FOLD=$fold PUBLIC_STAGE2_ARCH=$arch
    python 05_train_stage2_graph.py && python 06_train_stage3_gate.py && python 07_evaluate.py || { echo "failed: $arch fold $fold"; exit 1; }
  done
done
unset PUBLIC_STAGE2_ARCH PUBLIC_FOLD

export PUBLIC_POOL_FOLDS="${FOLDS// /,}"
export PUBLIC_COMPARE_ARMS="${ARCHS// /,}"
python 13_compare_graph_vs_temporal.py
