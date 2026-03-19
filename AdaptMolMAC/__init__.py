"""AdaptMolMAC public package interface."""

from .config import Settings
from .err import ErrorDetect
from .mcutils import BinaryUtils, CodebookGenerator, Logger, Plotter, generate_preamble_bits, logger
from .models import ChanneModel_Rx, ChannelModel_Rx, ChannelModel_Tx, yRxData
from .processing import DynamicDecoder, SignalProcessor, StationaryProcessor
from .sim import simParams
from .viterbi import convolutional_encode, hypothesis_extended_viterbi_decode, viterbi_decode

__version__ = "0.1.0"

__all__ = [
    "BinaryUtils",
    "ChanneModel_Rx",
    "ChannelModel_Rx",
    "ChannelModel_Tx",
    "CodebookGenerator",
    "DynamicDecoder",
    "ErrorDetect",
    "Logger",
    "Plotter",
    "Settings",
    "SignalProcessor",
    "StationaryProcessor",
    "convolutional_encode",
    "generate_preamble_bits",
    "hypothesis_extended_viterbi_decode",
    "logger",
    "simParams",
    "viterbi_decode",
    "yRxData",
]
