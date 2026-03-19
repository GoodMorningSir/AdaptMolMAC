"""Plotting helpers for waveform comparison and debugging.

This module centralizes small visualization utilities used across decoding and
estimation code.
"""

import matplotlib.pyplot as plt
class Plotter:
    """Small plotting helpers used by decoding and debugging routines.

    The class intentionally exposes compact static helpers so callers can build
    quick visual diagnostics without duplicating plotting boilerplate.
    """

    @staticmethod
    def draw_originPic_predictPic(origin_signal=None, predict_signal=None, Point_list=[]):
        """Plot original and predicted signals on the same figure.

        Args:
            origin_signal (array-like | yRxData | None): Reference waveform.
            predict_signal (array-like | yRxData | None): Predicted waveform.
            Point_list (list[int]): Vertical marker positions to highlight.

        Returns:
            matplotlib.figure.Figure: Figure containing the comparison plot.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        assert(origin_signal is not None or predict_signal is not None)

        if hasattr(origin_signal, 'yRxData'):
            origin_signal = origin_signal.yRxData
        if hasattr(predict_signal, 'yRxData'):
            predict_signal = predict_signal.yRxData

        if origin_signal is not None:
            ax.plot(origin_signal, label='Decode Sequence', color='blue')
        if predict_signal is not None:
            ax.plot(predict_signal, label='Predict Sequence', color='orange', alpha=0.7)
        for point in Point_list:
            ax.axvline(x=point, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Decode Sequence vs Predict Sequence')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)
        return fig
