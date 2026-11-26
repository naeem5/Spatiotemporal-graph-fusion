# NCM460

The same three stages on the second cell line. All settings are in `config.py`; set `NCM_DATA_ROOT` to the folder that contains `ncm460-p1/` (TIFF recordings) and run from this folder:

```bash
python 01_build_index.py            # index of the recordings
python 01e_cache_graph_inputs.py    # z-projections for Stage 2 and 3
python 01f_cache_stage1_frames.py   # middle frames for Stage 1
python 02_train_stage1_appearance.py
python 03_train_stage2_graph.py
python 04_train_stage3_gate.py      # one run per seed
python 05_evaluate.py               # validation and test results
```
