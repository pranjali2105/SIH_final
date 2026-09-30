# App integration checklist: `tft_ep2.onnx`

The full contract (input layout, feature order, normalisation stats, EKF and map-matching config) is in `tft_ep2_manifest.json`. Work through this list before shipping the model in the app.

## 1. ONNX Runtime session (blocking)
- [ ] Create the session with **`ORT_ENABLE_BASIC`** graph optimisation. With `ORT_ENABLE_EXTENDED` or `ORT_ENABLE_ALL` (the default), onnxruntime's operator fusions compute this model **wrongly**: differences up to 0.6 on real windows (verified with onnxruntime 1.29). BASIC matches PyTorch to about 1e-6.
  - Android (Java/Kotlin): `opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)`
  - iOS/ObjC: `[options setGraphOptimizationLevel:ORTGraphOptimizationLevelBasic error:&err]`
  - Python: `so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC` (see `idr/onnx_runtime.py`)
- [ ] **Startup assertion:** read the level back from the session options and fail fast if it isn't BASIC.
- [ ] **Startup self-check:** run the golden windows in `tft_ep2_golden.npz` (4 real validation windows: braking, roundabout, constant speed and random). Every output must satisfy `max |onnx − golden| < 1e-4`. If it doesn't, disable the model and fall back to plain IMU dead reckoning. The reference implementation is `idr/onnx_runtime.make_session`.
- [ ] Use the CPU execution provider. NNAPI, CoreML and GPU providers aren't verified; if you use one, repeat the golden check.
- [ ] Don't re-optimise or quantise the model offline without re-running `python -m pytest idr/tests/test_onnx_parity.py`.

## 2. Inputs
- [ ] Sampling is 10 Hz. `seq` is (B, 650, 17): 5 s of context with GNSS, then 60 s of outage. Use the feature order from `manifest.inputs.seq.feature_order`.
- [ ] Z-score the first 8 channels with `manifest.normalisation.norm_stats` and clip to ±6σ.
- [ ] Zero-pad the future (not yet observed) steps of the outage.
- [ ] Apply the phone axis calibration (`idr/normalise.calibrate`) and gyro bias removal before building `u_raw`.
- [ ] Make heading features relative to the last GNSS heading ψ0.
- [ ] `static` contains v0/30 and the destination features. Set `route_available` to 0 when there is no route.
- [ ] `route` holds 100 samples spaced 20 m apart: sin and cos of (bearing − ψ0), curvature×20, and the valid flag.

## 3. Downstream (unchanged, validated configuration)
- [ ] Integrate `u_hat` and `p_stop` in the 9-state EKF (`idr/nav_filter.py`, with `TFT_CFG` sig_w 0.05 and sig_a 0.5). Speed calibration stays **off**.
- [ ] Use the map matcher with the `MMConfig()` defaults: full-state map updates, roundabout mode off (as run by `run_outage` defaults), landmark scoring off, and the Experiment A `along_landmark` **off** unless it is promoted.
- [ ] Record the OSM extract version the app ships (`manifest.osm_cache`).

## 4. Release gate
- [ ] `python -m pytest idr/tests/test_onnx_parity.py` passes in CI. It compares the PyTorch checkpoint with the ONNX model on random inputs and real golden windows, and checks batch independence.
- [ ] The manifest SHA-256 of the `.onnx` file matches the file shipped in the app bundle.
