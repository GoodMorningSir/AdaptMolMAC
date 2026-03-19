"""Shared configuration constants for AdaptMolMAC.

The values in this module define package-wide defaults for signal framing,
anomaly detection, channel estimation, logging, and decoding performance.
"""

class Settings:
    """Central configuration values shared by the package."""

    # Default signal-end index. It is rescaled when the simulation time base
    # changes.
    SIG_END = 31

    # Extra candidate peaks retained beyond the baseline preamble peak count.
    PEAK_POINT_EXCE_CUT = 1

    # Default number of preamble bits used by transmitters and processors.
    PREAMBLE_NUM = 3

    # Default anomaly probability threshold for probabilistic detection.
    DEFAULT_PROB_THRESHOLD = 1e-4

    # Probability level treated as a confident anomaly during decoding.
    SURE_ERR_PROB_THRESHOLD = 1e-6

    # Default number of extra bits explored when extending a new-signal
    # hypothesis.
    DEFAULT_EXTENDED_BITS_NUM = 3

    # Enables the fine-tuning stage after coarse channel fitting.
    FINE_TUNE_ENABLE = True

    # Allowed preamble error rate for each supported detected-signal count.
    ALLOW_BITS_ERRORPEC_IN_DETECT = [0.4, 0.4, 0.4, 0.4]

    # Accumulated score required to trigger new-signal detection.
    NEW_SIGNAL_DETECT_THRESHOLD = 2

    # Recovery amount subtracted from the detection score after normal
    # observations.
    NEW_SIGNAL_DETECT_RECOVERY = 0.5

    # Fixed synchronization/check pattern inserted between preamble and payload.
    CHECK_CODE = "101010"
    
    # Default switch for runtime logging output.
    LOG_INIT_ENABLED = False

    # Default switch for saving logs to files.
    LOG_FILES_SAVE_ENABLED = False
    
    # Maximum worker count used for multiprocessing during hypothesis
    # evaluation.
    PROCESSING_MULTI_PROCESSING_NUM = 8

