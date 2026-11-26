# Public datasets

The three stages trained from scratch on LCTP, PASTIS and HyperKvasir. Run the scripts from this folder; the dataset and protocol are chosen with environment variables read by `config.py`.

| Dataset | `PUBLIC_DATASET` | `PUBLIC_PROTOCOL` (folds) | Source (licence) |
|---|---|---|---|
| LCTP, phase-contrast time-lapse of A549 cells | `lctp_a549` | `plate` (0-2), `compound` (0-3) | Figshare, DOI 10.17044/scilifelab.29315642 (CC BY-NC-SA 4.0) |
| PASTIS, Sentinel-2 time series | `pastis` | `official5` (0-4) | Zenodo, DOI 10.5281/zenodo.5012942 (CC BY 4.0) |
| HyperKvasir, endoscopy images | `hyperkvasir` | `official5` (0-4) | https://datasets.simula.no/hyper-kvasir/ (CC BY 4.0) |

```bash
export PUBLIC_DATASET=lctp_a549 PUBLIC_LCTP_ROOT=/path/to/lctp PUBLIC_PROTOCOL=plate
python 01_build_index_lctp.py                       # once; 02_ for HyperKvasir (PUBLIC_HK_ROOT), 03_ for PASTIS (PUBLIC_PASTIS_ROOT)
for fold in 0 1 2; do
  export PUBLIC_FOLD=$fold
  python 04_train_stage1_appearance.py && python 05_train_stage2_graph.py && python 06_train_stage3_gate.py && python 07_evaluate.py
done
PUBLIC_POOL_FOLDS=0,1,2 python 08_pool_lctp.py      # pooled result; 09_pool_official_folds.py for PASTIS and HyperKvasir
```

Other variables: `PUBLIC_FRAMES=anchor` (graph expert reads only the anchor frame, single-frame test), `PUBLIC_STAGE2_ARCH=temporal|temporal_nodes` (control experts without spatial edges),
`PUBLIC_OUT_DIR`, `PUBLIC_TILE_CACHE` (output and cache folders), `PUBLIC_GPU`. The other numbered scripts work on the stored test outputs: `10_compare_single_vs_all_frames.py`
(single frame versus all frames), `11_top_k_accuracy.py`, `12_per_fold_gains.py`, and `13_compare_graph_vs_temporal.py`, which compares the patch graph with the control experts
trained by `bash run_temporal_control.sh`.
