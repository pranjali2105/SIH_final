# SIH 2026 PS 168: smartphone dead reckoning during GNSS outages

When a phone loses GNSS, this pipeline estimates the vehicle's position from the phone's IMU:

- a **residual Temporal Fusion Transformer** corrects the IMU readings;
- a **9-state EKF** with an adaptive non-holonomic constraint integrates them;
- an **OSM particle-filter map matcher** keeps the estimate on real roads.

**Results.** Median 60 s drift (end-point error ÷ distance travelled), 3 seeds:

| Split | Median drift |
|---|---|
| Held-out Y1 + M | **9.6%** |
| Held-out Vw4 | **9.1%** |
| Validation S2 + Vtb5 | 9.4% |

The target is under 10%. For comparison, raw phone IMU with physics integration alone gives 28.5%.

The full write-up, with architecture, method, drift analysis and experiment log, is in [`docs/PS168_Dead_Reckoning_Report.pdf`](docs/PS168_Dead_Reckoning_Report.pdf) (HTML version: `docs/report.html`).

## Layout
```
idr/
  data.py, normalise.py        load IO-VNBD, phone/vehicle clock resync, phone-axis calibration, normalisation
  dataset.py, route_features.py 60 s outage windows, random destination + route features
  model.py, losses.py, train.py ResidualTFT (VSN -> causal CNN -> LSTM -> causal attention -> route cross-attention)
  physics.py                   differentiable integration (same equations in training and in the app)
  nav_filter.py                9-state EKF, adaptive NHC, delayed updates with replay
  osm_map.py, map_matching.py  OSM road graph, particle-filter map matching, turn landmarks
  confidence.py                confidence layer for map corrections (experimental, off by default)
  evaluate.py, evaluate_mm.py  open-loop and map-aided evaluation (3 seeds)
  speed_slices.py, tail_diag.py, longitudinal_diag.py, expA_compare.py   error diagnostics
  export.py, onnx_runtime.py   ONNX export, manifest, startup self-check
  tests/test_onnx_parity.py    CI: PyTorch vs ONNX parity
  artifacts/                   checkpoints (tft_ep2.pt = validated model), evaluation JSONs, logs
  export/                      tft_ep2.onnx, manifest, golden windows, APP_INTEGRATION.md
  figures/                     plots
docs/                          report (PDF + HTML)
```

## Data (not in this repo)
- The dataset is IO-VNBD: <https://github.com/onyekpeu/IO-VNBD>. Clone it to `IO-VNBD/` and put the LFS data under `raw/`.
- `idr/cache/` (parquet and OSM caches) is rebuilt automatically on the first run.

## Usage
```bash
pip install -r requirements.txt
python -m idr.train --help                                                        # training
python -m idr.evaluate_mm --tag tft_ep2 --drives Y1 M --seeds 3 --core           # map-aided evaluation
python -m idr.map_matching                                                        # filter self-tests
python -m pytest idr/tests/test_onnx_parity.py                                    # ONNX parity
```

## Deployment note
The ONNX model **must** run with `ORT_ENABLE_BASIC` graph optimisation. The higher levels contain an onnxruntime 1.29 fusion bug that corrupts this model's outputs. See `idr/export/APP_INTEGRATION.md`.
