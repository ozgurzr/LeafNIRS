# Phase 1 Development Notes

## Decisions Made

### Dual Loader Architecture

Both `snirf` library (Method A) and raw `h5py` (Method B) were implemented behind a common `SNIRFLoaderBase` interface. Results on real data (ds007420):

- **Method B is 2x faster** (~93ms vs ~187ms)
- Both produce **identical outputs** (verified by `np.allclose`)
- Method B gives full parsing control; Method A has built-in SNIRF validation

The GUI lets users switch loaders via **View → SNIRF Loader**.

### Real Data

Tested against **OpenNeuro ds007420** — a multi-session fNIRS dataset with:

- Resting, BallSqueezing, and Motion tasks
- 14 sources, 32 detectors, 2 wavelengths (760/850 nm)
- 200 channels per recording, ~300s duration, 8.72 Hz sampling

### Bug Fix: h5py Scalar Parsing

Real SNIRF files store scalars like `sourceIndex` as shape `(1,)` numpy arrays. Added `_unwrap_scalar()` to convert these to plain Python types.

### Performance Optimization

Initial graph widget was extremely laggy with 200 channels. Fixed by:

1. Disabling anti-aliasing (biggest impact)
2. Adding `clipToView=True` and `downsample` to PlotDataItems
3. Setting empty data `[]` on hidden channels (zero GPU cost)
4. Reducing default visible channels

### Channel Quality

CV-based quality metric:

- CV < 0.1% → **Flat** (no contact)
- CV > 50% → **Noisy** (bad coupling or motion)
- Otherwise → **OK**

## Test Suite

30 tests + 10 subtests covering:

- Individual loader correctness
- Cross-loader comparison
- Multi-file loading (different tasks/sessions)
- Performance benchmarks
- Error handling (missing file, wrong extension)
