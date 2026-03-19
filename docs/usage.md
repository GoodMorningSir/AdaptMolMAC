# AdaptMolMAC Usage Guide

This guide covers the main workflow, then the key modules: `config`, `models`, `processing`, `mcutils`, `err`, and `viterbi`. The best starting point is `examples/basic_pipeline.py`.

## Installation

```bash
pip install -e .
```

Common imports:

```python
from AdaptMolMAC import (
    Settings,
    ChannelModel_Tx,
    ChannelModel_Rx,
    yRxData,
    SignalProcessor,
    StationaryProcessor,
    DynamicDecoder,
    ErrorDetect,
    BinaryUtils,
    CodebookGenerator,
    Logger,
    Plotter,
    generate_preamble_bits,
    convolutional_encode,
    viterbi_decode,
    hypothesis_extended_viterbi_decode,
    logger,
)
```

## Task 1: Read `examples/basic_pipeline.py`

`examples/basic_pipeline.py` shows the full path from transmission to payload recovery. It creates two transmitters, generates two waveforms, adds them into one received signal, preprocesses the signal, estimates a receive-side channel template, runs dynamic decoding, then removes the preamble and check bits before final Viterbi decoding.

The script starts with a convolutional-code generator matrix:

```python
generator = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 0, 0],
]
```

Use the same generator matrix in transmission and decoding.

Two payload strings are then defined, and two `ChannelModel_Tx` objects are created. The key parameters are `interval` and `tx_offset`. `interval` sets symbol spacing. `tx_offset` sets where a transmitter starts on the global time axis. Different values create overlap between signals.

`ChannelModel_Tx.transmit(data)` does not send `data` directly. If `viterbi_gen` is set, it first calls `convolutional_encode()`. If `n_preamble` is set, it prepends the preamble from `generate_preamble_bits()` and the fixed pattern from `Settings.CHECK_CODE`. The transmitted sequence is:

```text
preamble + check code + convolutionally encoded payload
```

The example uses `yRxData()` as the receive buffer:

```python
y_rx0 = amm.yRxData()
y_rx0 += tx1.transmit(tx1_bits)
y_rx0 += tx2.transmit(tx2_bits)
```

`yRxData` is the shared signal container used by simulation, preprocessing, and decoding. Adding two `yRxData` objects overlays their raw waveforms sample by sample.

Preprocessing:

```python
signal_processor = amm.SignalProcessor()
y_rx = signal_processor.process_signal(y_rx0, x_offset=1)
filtered_y_rx = signal_processor.kalman_filter(y_rx)
filtered_y_rx = signal_processor.adaptive_threshold_filter(filtered_y_rx)
filtered_y_rx = signal_processor.retain_peak_points_filter(
    peak_points_num=3 + amm.Settings.PEAK_POINT_EXCE_CUT,
    yrx_data=filtered_y_rx,
)
```

`process_signal()` estimates the signal start and removes the baseline. `kalman_filter()` smooths the waveform. `adaptive_threshold_filter()` removes weak values. `retain_peak_points_filter()` crops the front part of the waveform so channel estimation sees the useful peak region.

Channel estimation:

```python
stationary_processor = amm.StationaryProcessor(
    n_preamble=amm.Settings.PREAMBLE_NUM
)
channel_info, mse = stationary_processor.estimate_channel(
    y_rx,
    filtered_y_rx,
    interval_dev=0,
    start_pos_dev=1,
)
channel_info.reset_start_pos()
```

`estimate_channel()` detects peaks, matches the triangular preamble structure, creates a `ChannelModel_Rx`, fits waveform key points, and returns the fitted channel plus its MSE.

Dynamic decoding:

```python
decoder = amm.DynamicDecoder(channel_info.yRx, [channel_info], generator)
decoded_channels = decoder.decode(MAX_SIGNAL_NUM=2)
```

`DynamicDecoder` tracks one or more active channels over time. It estimates the next bit for each channel, checks residual error, and can trigger new-signal detection when the current channel set no longer explains the waveform.

Final payload recovery:

```python
preamble_len = len(stationary_processor.preamble) + len(amm.Settings.CHECK_CODE)
encoded_payload = decoded_channels[0][preamble_len:len(tx1.encode_data[0])]
decoded_payload = amm.viterbi_decode(
    encoded_payload,
    generator,
    len(generator[0]),
)
```

Do not pass the full decoded channel string into `viterbi_decode()`. Pass only the encoded payload part.

Minimal single-channel example:

```python
import AdaptMolMAC as amm

generator = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 0, 0],
]

tx = amm.ChannelModel_Tx(
    interval=17,
    tx_offset=50,
    amplitude=1.0,
    n_preamble=amm.Settings.PREAMBLE_NUM,
    viterbi_gen=generator,
)

y_rx = tx.transmit("11001010101")

processor = amm.SignalProcessor()
processed = processor.process_signal(y_rx, x_offset=1)
processed = processor.kalman_filter(processed)
processed = processor.adaptive_threshold_filter(processed)

stationary = amm.StationaryProcessor(n_preamble=amm.Settings.PREAMBLE_NUM)
channel_info, mse = stationary.estimate_channel(y_rx, processed)
channel_info.reset_start_pos()

decoder = amm.DynamicDecoder(channel_info.yRx, [channel_info], generator)
decoded_channels = decoder.decode(MAX_SIGNAL_NUM=1)

preamble_len = len(stationary.preamble) + len(amm.Settings.CHECK_CODE)
decoded_payload = amm.viterbi_decode(
    decoded_channels[0][preamble_len:len(tx.encode_data[0])],
    generator,
    len(generator[0]),
)
```

## Task 2: `config.py` and `Settings`

`Settings` stores package-wide constants. Read it directly as a class.

Main fields:

- `SIG_END`: default signal-end index used in waveform sizing.
- `PEAK_POINT_EXCE_CUT`: extra peaks kept beyond the baseline preamble count.
- `PREAMBLE_NUM`: default preamble order.
- `DEFAULT_PROB_THRESHOLD`: default anomaly threshold for probabilistic detection.
- `SURE_ERR_PROB_THRESHOLD`: threshold for strong anomaly decisions.
- `DEFAULT_EXTENDED_BITS_NUM`: default extension length for hypothesis search.
- `FINE_TUNE_ENABLE`: enable fine tuning after coarse channel fitting.
- `ALLOW_BITS_ERRORPEC_IN_DETECT`: allowed preamble error rate by detected signal count.
- `NEW_SIGNAL_DETECT_THRESHOLD`: score needed to trigger new-signal detection.
- `NEW_SIGNAL_DETECT_RECOVERY`: recovery amount subtracted after normal observations.
- `CHECK_CODE`: fixed check pattern inserted after the preamble.
- `LOG_INIT_ENABLED`: default runtime logging switch.
- `LOG_FILES_SAVE_ENABLED`: default file-save switch for logs.
- `PROCESSING_MULTI_PROCESSING_NUM`: worker count for multiprocessing search.

Typical usage:

```python
import AdaptMolMAC as amm

print(amm.Settings.PREAMBLE_NUM)
print(amm.Settings.CHECK_CODE)

amm.Settings.PREAMBLE_NUM = 4
amm.Settings.FINE_TUNE_ENABLE = False
```

Change settings before creating transmitters and processors. They are global within the current Python process.

## Task 3: `models`

The `models` layer contains the signal container, the transmit model, and the fitted receive model.

### `yRxData`

Constructor:

```python
yRxData(data=None, send_bits=None, process_data=None, ifLogger=True)
```

Use `data` for raw samples. Use `process_data` for already processed samples.

Key members:

- `raw_data`: returns raw noiseless data. Unavailable after locking.
- `yRxData`: returns processed data. Noise is injected lazily if needed.
- `lock()`: freezes processed data and discards raw data.
- `visualize()`: plots the waveform and associated transmitted bits.
- `__add__(other)`: overlays two unlocked `yRxData` objects.

Example:

```python
import AdaptMolMAC as amm

y1 = amm.yRxData(data=[0, 1, 2, 1, 0], send_bits=["101"])
y2 = amm.yRxData(data=[0, 0.5, 1.0, 0.5, 0], send_bits=["111"])
y_sum = y1 + y2
samples = y_sum.yRxData
y_sum.lock()
```

### `ChannelModel_Tx`

Constructor:

```python
ChannelModel_Tx(
    interval,
    tx_offset,
    amplitude=1.0,
    n_preamble=None,
    viterbi_gen=None,
    ChannelParam=None,
)
```

Main methods:

- `set_noise_param(noiseb, noisen, noisep)`: updates global noise settings.
- `transmit(data)`: encodes, prepends preamble and check code if configured, simulates the waveform, and returns `yRxData`.
- `parameters`: returns a dictionary with `interval` and `tx_offset`.

Example:

```python
import AdaptMolMAC as amm

generator = [[1, 1, 1], [1, 0, 1], [1, 0, 0]]
amm.ChannelModel_Tx.set_noise_param(noiseb=0.1, noisen=1.0, noisep=0.01)

tx = amm.ChannelModel_Tx(
    interval=20,
    tx_offset=120,
    amplitude=1.0,
    n_preamble=amm.Settings.PREAMBLE_NUM,
    viterbi_gen=generator,
)

y_rx = tx.transmit("101011001")
print(tx.transmit_data[-1])
print(tx.encode_data[-1])
print(tx.parameters)
```

### `ChannelModel_Rx`

Source class name: `ChanneModel_Rx`. Public alias: `ChannelModel_Rx`.

Constructor:

```python
ChannelModel_Rx(interval, peak1_x, yRx, n_preamble, n_peak, sig_end=None)
```

Most users get it from `StationaryProcessor.estimate_channel()`.

Main methods:

- `fresh_key1_value()`: refreshes the first key-point amplitude.
- `cal_key1_value(x)`: estimates the first key-point amplitude from preamble peaks.
- `visualize_bit(bit_value=1)`: plots a single-bit template.
- `train_keypoint(interval_dev=0, start_pos_dev=0)`: fits waveform key points by search.
- `fine_tune_params(delta_start=5, delta_interval=5)`: refines interval and start position.
- `generate_bit_waveform(bit_value)`: generates a single-bit waveform.
- `generate_wave_prediction(bits_sequence)`: synthesizes the waveform for a bit sequence.
- `generate_preamble()`: synthesizes the preamble waveform.
- `reset_start_pos()`: trims cached signal after a positive start offset.
- `visualize()`: plots the fitted preamble against the received signal.

Example:

```python
channel_info, mse = stationary.estimate_channel(y_rx, filtered_y_rx)
bit1_wave = channel_info.generate_bit_waveform(1)
pred = channel_info.generate_wave_prediction("1100101")
fig = channel_info.visualize()
channel_info.reset_start_pos()
```

## Task 4: `processing`

This module contains preprocessing, stationary-point detection, channel fitting, and dynamic decoding.

### `SignalProcessor`

Constructor:

```python
SignalProcessor(kl_threshold=2.0)
```

Main methods:

- `process_signal(yrx_data, x_offset=1)`: detects the signal start and removes baseline.
- `kalman_filter(...)`: smooths a 1D signal with a Kalman filter.
- `percentile_threshold_filter(yrx_data, peek_pct=95.0, threshold_pct=0.5)`: zeros samples below a percentile-based threshold.
- `adaptive_threshold_filter(yrx_data)`: zeros samples below an Otsu threshold.
- `retain_peak_points_filter(peak_points_num, yrx_data)`: keeps the signal up to a selected peak index.

Typical chain:

```python
processor = amm.SignalProcessor()
processed = processor.process_signal(y_rx, x_offset=1)
processed = processor.kalman_filter(processed)
processed = processor.adaptive_threshold_filter(processed)
processed = processor.retain_peak_points_filter(
    peak_points_num=amm.Settings.PREAMBLE_NUM + amm.Settings.PEAK_POINT_EXCE_CUT,
    yrx_data=processed,
)
```

### `StationaryPoints`

Lightweight container for detected peaks and valleys.

Main methods:

- `extract_peaks()`
- `extract_valleys()`
- `sort_by_position()`
- `append_peak(point)`
- `append_valley(point)`

### `StationaryProcessor`

Constructor:

```python
StationaryProcessor(n_preamble, d2_threshold=0, n_peak=None)
```

Main methods:

- `detect(y_smooth)`: detects peaks and valleys from a smoothed signal.
- `compute_d_s(sum_1, sum_T, sum_T2, sum_k, sum_kT)`: solves the timing model.
- `gen_N(d, s, n)`: builds the ideal triangular timing sequence.
- `compute_sequence(K, n)`: fits one candidate sequence.
- `non_continuous_matching(K, n)`: finds the best subset matching the preamble pattern.
- `estimate_channel(yRx, filtered_yRx=None, interval_dev=0, start_pos_dev=0, visualize=False)`: estimates a `ChannelModel_Rx` and returns `(channel, mse)`.

Example:

```python
stationary = amm.StationaryProcessor(n_preamble=amm.Settings.PREAMBLE_NUM)
channel_info, mse = stationary.estimate_channel(
    y_rx,
    filtered_y_rx,
    interval_dev=0,
    start_pos_dev=1,
)
```

### `ChannelPredictionItem`

Internal state object used by `DynamicDecoder`.

Main methods and properties:

- `reset_abnormal_detector()`
- `generate_bit_waveform(bit)`
- `add_err(error)`
- `detect_err(error)`
- `update_bit(bit)`
- `generate_waveform_from_bits(bit_sequence)`
- `bits`
- `pred`
- `pred_end`
- `interval`
- `preamble_bits`

### `DynamicDecoder`

Constructor:

```python
DynamicDecoder(yRx, chnl_info_list, generator=None)
```

Main methods:

- `decode(MAX_SIGNAL_NUM=None, visualize=False)`: runs the full dynamic decoding loop.
- `draw_signal(visualize=False)`: logs or plots the current reconstruction.
- `save_state(time_ptr, permanent=False, tag=None)`: saves decoder state.
- `restore_state(state_id)`: restores a saved state.
- `delete_state(state_id)`: deletes a saved state.

Internal routines:

- `_estimate_bits()`
- `_estimate_bit_in_chnl_list()`
- `_detect_new_signal()`
- `_detect_new_signal_single_process()`
- `_detect_new_signal_multi_process()`
- `_evaluate_combination()`
- `_process_detection_result()`

High-level behavior:

1. Predict the next bit for each active channel.
2. Measure reconstruction error.
3. Detect persistent anomalies.
4. Try to explain the residual with a new channel.
5. Roll back if the new channel fails the preamble check.

Example:

```python
decoder = amm.DynamicDecoder(channel_info.yRx, [channel_info], generator)
decoded_channels = decoder.decode(MAX_SIGNAL_NUM=2, visualize=False)
decoder.draw_signal(visualize=False)
```

## Task 5: `mcutils`

This package contains shared utilities for bits, preambles, codebooks, anomaly detection, logging, and plotting.

### `BinaryUtils`

Static helpers:

- `validate_binary_string(input_str)`
- `validate_binary_list(input_list)`
- `binary_string_to_list(input_str)`
- `list_to_binary_string(input_list)`

### `generate_preamble_bits(n)`

Generates the triangular-structure preamble used by both the transmitter and the receiver.

Example:

```python
from AdaptMolMAC import BinaryUtils, generate_preamble_bits

assert BinaryUtils.validate_binary_string("10101")
bits_list = BinaryUtils.binary_string_to_list("10101")
bits_str = BinaryUtils.list_to_binary_string([1, 0, 1, 0, 1])
preamble = generate_preamble_bits(3)
```

### `CodebookGenerator`

Constructor:

```python
CodebookGenerator(capacity)
```

Main methods:

- `validate(code)`
- `get_code(id)`
- `get_id(code)`
- `__str__()`

Example:

```python
from AdaptMolMAC import CodebookGenerator

codebook = CodebookGenerator(4)
print(codebook.get_code(0))
print(codebook.get_id(codebook.get_code(0)))
print(codebook.validate("000"))
```

### `TProbAnomalyDetector`

Constructor:

```python
TProbAnomalyDetector(
    yRx=None,
    min_capacity=3,
    prob_threshold=Settings.DEFAULT_PROB_THRESHOLD,
    window_size=None,
)
```

Main methods:

- `reset()`
- `detect(x, if_add=False)`
- `add(x)`
- `get_statistics()`

Used in `SignalProcessor.process_signal()` and channel-wise error monitoring inside `DynamicDecoder`.

### `KLStreamingAnomalyDetector`

Constructor:

```python
KLStreamingAnomalyDetector(base_sequence=None, interpolation_factor=10)
```

Main methods:

- `detect_anomaly(kl_threshold=2.1, verbose=False)`
- `remove_original_point(original_pos)`
- `append(new_item)`
- `original_sequence_with_x()`

This detector is not part of the default pipeline, but it is useful for experiments.

### `Logger`

The package exposes a global logger instance as `logger`.

Main class methods:

- `Logger.set_enabled(enabled=True)`
- `Logger.enable()`
- `Logger.disable()`

Main instance methods:

- `log(data, description="", predict_signal=None, point_list=[], text_save=False, level="INFO", img_save_subfolder=None)`
- `log_image(fig, description, img_save_subfolder=None, img_name=None, FIG_CLOSE=True)`
- `save_temp_file(data, img_save_subfolder="temp_files")`

Example:

```python
import AdaptMolMAC as amm

amm.Logger.enable()
amm.logger.log("experiment start")
amm.logger.log({"interval": 17, "offset": 50})
amm.logger.log([0, 1, 2, 1, 0], description="demo signal")
```

### `Plotter`

Main helper:

- `Plotter.draw_originPic_predictPic(origin_signal=None, predict_signal=None, Point_list=[])`

Used to compare measured and predicted waveforms.

## Task 6: `err`

The `err` package currently exposes `ErrorDetect`.

Constructor:

```python
ErrorDetect(Seed, *args)
```

Pass bit strings in alternating send/receive order:

```python
from AdaptMolMAC import ErrorDetect

detector = ErrorDetect(
    20260320,
    "01010011010", "01010011010",
    "10010011011", "10010001011",
)
```

Main method:

- `compare_accuracy()`: returns per-pair accuracy and appends the result to `result.csv`.

Internal helpers:

- `_to_str(bits)`
- `_calc_acc(send_bits, recv_bits)`

If the received string is shorter than the sent string, the reported accuracy is `0.0`.

## Task 7: `viterbi`

This module handles convolutional encoding, classical Viterbi decoding, path visualization, and partial-sequence hypothesis extension.

Main functions:

- `validate_viterbi_generator(viterbi_gen)`: validates the generator matrix and returns the constraint length.
- `convolutional_encode(input_bits, generators, constraint_length)`: encodes a payload bit string.
- `viterbi_decode(encoded, generators, constraint_length)`: decodes an encoded payload bit string.
- `visualize_path_metrics(path_metrics_history, path_history)`: plots decoder state transitions.
- `flip_bit(encoded, N)`: flips random bits in a mutable encoded sequence.
- `hypothesis_extended_viterbi_decode(encoded_bits_with_preamble, generators, extend_decode_num=3, premble_num=3)`: generates candidate extensions for partially decoded sequences.

Example:

```python
from AdaptMolMAC import (
    validate_viterbi_generator,
    convolutional_encode,
    viterbi_decode,
    hypothesis_extended_viterbi_decode,
)

generator = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 0, 0],
]

constraint_length = validate_viterbi_generator(generator)
encoded = convolutional_encode("1100101", generator, constraint_length)
decoded = viterbi_decode(encoded, generator, constraint_length)
extended = hypothesis_extended_viterbi_decode(
    "110110010101010" + encoded,
    generator,
    extend_decode_num=3,
    premble_num=3,
)
```

Use `viterbi_decode()` on the encoded payload only. Do not include the preamble or `Settings.CHECK_CODE`.

## Suggested Reading Order

Read the code in this order:

1. `examples/basic_pipeline.py`
2. `AdaptMolMAC/config.py`
3. `AdaptMolMAC/models/channel_model.py`
4. `AdaptMolMAC/processing/signal_processor.py`
5. `AdaptMolMAC/processing/stationary_detector.py`
6. `AdaptMolMAC/processing/dynamic_decoder.py`
7. `AdaptMolMAC/mcutils/*`
8. `AdaptMolMAC/err/error_detect.py`
9. `AdaptMolMAC/viterbi/viterbi.py`

## Practical Notes

Keep three things consistent:

1. The same preamble rule on both transmit and receive sides.
2. The same generator matrix for encoding and decoding.
3. The correct split point between preamble/check bits and encoded payload.

If those three are correct, most remaining issues are tuning issues. If not, decoding quality will drop quickly.

