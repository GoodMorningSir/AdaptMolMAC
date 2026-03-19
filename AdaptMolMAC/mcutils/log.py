"""Logging helpers for text, figures, and intermediate signal artifacts.

The logger is intentionally lightweight and can be disabled globally through
package settings.
"""

import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import matplotlib
import numpy as np
import pandas as pd
from ..config import Settings

class Logger:
    """Lightweight logger for text, figures, and intermediate signal arrays.

    Attributes:
        LOG_ENABLED (bool): Global switch controlling logger activation.
        FILES_SAVE_ENABLED (bool): Whether CSV helper artifacts are stored.
    """

    LOG_ENABLED = Settings.LOG_INIT_ENABLED
    FILES_SAVE_ENABLED = Settings.LOG_FILES_SAVE_ENABLED
    
    def __init__(self, base_log_dir="log", log_name=None, img_folder="img"):
        """Prepare a timestamped logging workspace lazily.

        Args:
            base_log_dir (str): Base directory for log outputs.
            log_name (str | None): Optional prefix for the generated log folder.
            img_folder (str): Name of the image subdirectory.
        """
        self.base_log_dir = base_log_dir
        self.log_name = log_name
        self.img_folder = img_folder
        self.log_folder = None
        self.log_path = None
        self.default_img_folder = None
        self._workspace_ready = False

    @classmethod
    def set_enabled(cls, enabled=True):
        """Enable or disable runtime logging after logger creation."""
        cls.LOG_ENABLED = enabled

    @classmethod
    def enable(cls):
        """Enable runtime logging."""
        cls.set_enabled(True)

    @classmethod
    def disable(cls):
        """Disable runtime logging."""
        cls.set_enabled(False)

    def _ensure_workspace(self):
        """Create the logging workspace on first real output."""
        if self._workspace_ready:
            return

        os.makedirs(self.base_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.log_name:
            self.log_folder = os.path.join(self.base_log_dir, f"{self.log_name}_{timestamp}")
        else:
            self.log_folder = os.path.join(self.base_log_dir, f"log_{timestamp}")
        self.log_path = os.path.join(self.log_folder, "app.log")
        self.default_img_folder = os.path.join(self.log_folder, self.img_folder)
        os.makedirs(self.log_folder, exist_ok=True)
        os.makedirs(self.default_img_folder, exist_ok=True)
        self._workspace_ready = True

        if Logger.LOG_ENABLED:
            self._log_text(f"Log folder created: {self.log_folder}", "INFO")
            self._log_text(f"Log file path: {self.log_path}", "INFO")
            self._log_text(f"Default image storage path: {self.default_img_folder}", "INFO")

    def _log_text(self, content, level="INFO"):
        """Append one text entry to the log file.

        Args:
            content (str): Text payload to write.
            level (str): Log severity label.
        """
        if not Logger.LOG_ENABLED:
            return
        self._ensure_workspace()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(self.log_path, "a") as f:
            f.write(f"[{timestamp}] [{level}] {content}\n")

    def _log_dict(self, data, level="INFO"):
        """Append one dictionary entry in JSON form.

        Args:
            data (dict): Mapping to serialize.
            level (str): Log severity label.
        """
        if not Logger.LOG_ENABLED:
            return
        self._ensure_workspace()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(self.log_path, "a") as f:
            f.write(f"[{timestamp}] [{level}] JSON_DATA: {json.dumps(data)}\n")
            
    def save_temp_file(self, data, img_save_subfolder="temp_files"):
        """Persist raw numeric data as a CSV helper artifact.

        Args:
            data (object): Array-like data or signal container to store.
            img_save_subfolder (str): Subdirectory for the saved CSV file.
        """
        if not Logger.FILES_SAVE_ENABLED:
            return
        self._ensure_workspace()
        while hasattr(data, 'yRxData'):
            data = data.yRxData
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        save_dir = os.path.join(self.log_folder, img_save_subfolder)
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"data_{timestamp}.csv")

        if isinstance(data, np.ndarray):
            pd.DataFrame(data).to_csv(csv_path, index=False)
        elif isinstance(data, (list, tuple)):
            pd.DataFrame(list(data)).to_csv(csv_path, index=False)
        else:
            pd.DataFrame([data]).to_csv(csv_path, index=False)

        self._log_text(f"CSV_SAVED: {os.path.relpath(csv_path, self.log_folder)}", "INFO")
        
        
            
    def _log_list_as_image(self, data, description="", predict_signal=None, point_list=[], img_save_subfolder=None):
        """Render numeric sequences as plots and save the resulting figure.

        Args:
            data (object): Main sequence to visualize.
            description (str): Description recorded in the log.
            predict_signal (object | None): Optional secondary sequence.
            point_list (list[int]): Marker positions to show on the plot.
            img_save_subfolder (str | None): Optional subdirectory for output.

        Returns:
            str | None: Saved image path when logging is enabled.
        """
        if not Logger.LOG_ENABLED:
            return None
        
        while hasattr(data, 'yRxData'):
            data = data.yRxData
        while hasattr(predict_signal, 'yRxData'):
            predict_signal = predict_signal.yRxData
        
        fig = self._create_plot(
            origin_signal=data,
            predict_signal=predict_signal,
            point_list=point_list,
        )
        
        img_path = self.log_image(fig, description, img_save_subfolder=img_save_subfolder)
        return img_path


    def _create_plot(self, origin_signal=None, predict_signal=None, point_list=[]):
        """Create a standardized comparison plot for one or two signals.

        Args:
            origin_signal (array-like | None): Reference sequence.
            predict_signal (array-like | None): Predicted sequence.
            point_list (list[int]): Marker positions to overlay.

        Returns:
            matplotlib.figure.Figure: Figure containing the generated plot.
        """
        title = 'Origin Sequence vs Predict Sequence'
        figsize = (12, 6)
        origin_label = 'Origin Sequence'
        predict_label = 'Predict Sequence'
        
        fig = plt.figure(figsize=figsize)
        if origin_signal is not None and isinstance(origin_signal, list):
            origin_signal = origin_signal.tolist()
        if predict_signal is not None and isinstance(predict_signal, list):
            predict_signal = predict_signal.tolist()
            
        assert(origin_signal is not None or predict_signal is not None), "At least one signal must be provided"
        if origin_signal is not None:
            plt.plot(origin_signal, label=origin_label, color='blue')
        if predict_signal is not None:
            plt.plot(predict_signal, label=predict_label, color='orange', alpha=0.7)
        for point in point_list:
            plt.axvline(x=point, color='red', linestyle='--', alpha=0.5)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel( 'Amplitude')
        plt.legend()
        plt.grid(True)
        return fig
    
    @staticmethod
    def is_underlying_array(obj, depth=0, max_depth=10):
        """Return True when an object eventually unwraps to array-like data.

        Args:
            obj (object): Object to inspect.
            depth (int): Current recursion depth.
            max_depth (int): Maximum recursion depth allowed.

        Returns:
            bool: Whether the object is or contains an array-like payload.
        """
        if depth > max_depth:
            return False
        if isinstance(obj, (list, tuple, np.ndarray)):
            return True
        elif hasattr(obj, 'yRxData'):
            return Logger.is_underlying_array(obj.yRxData, depth+1, max_depth)
        return False

    def log(self, data, description="", predict_signal=None, point_list=[], text_save = False, level="INFO", img_save_subfolder=None):
        """Log text, dictionaries, arrays, or signal containers.

        Args:
            data (object): Value to log.
            description (str): Optional description used for image logs.
            predict_signal (object | None): Optional secondary signal.
            point_list (list[int]): Marker positions for plotting.
            text_save (bool): Whether to also dump array-like data as text.
            level (str): Log severity label.
            img_save_subfolder (str | None): Optional output subdirectory.

        Returns:
            str | None: Saved image path for array-like data, otherwise None.
        """
        if not Logger.LOG_ENABLED:
            return None
        if isinstance(data, str):
            self._log_text(data, level)
        elif isinstance(data, dict):
            self._log_dict(data, level)
        elif Logger.is_underlying_array(data):
            if text_save:
                self._log_text(f"LIST_DATA: {data}", level)
            return self._log_list_as_image(
                data, 
                description=description,
                predict_signal=predict_signal,
                point_list=point_list,
                img_save_subfolder=img_save_subfolder 
            )
        else:
            self._log_text(str(data), level)
        return None

    def log_image(self, fig, description, img_save_subfolder=None, img_name = None ,FIG_CLOSE=True):
        """Save a matplotlib figure and record its relative path.

        Args:
            fig (matplotlib.figure.Figure): Figure to save.
            description (str): Description written to the log.
            img_save_subfolder (str | None): Optional image subdirectory.
            img_name (str | None): Optional output filename.
            FIG_CLOSE (bool): Whether to close the figure after saving.

        Returns:
            str | None: Path to the saved image when logging is enabled.
        """
        if not Logger.LOG_ENABLED:
            return None
        self._ensure_workspace()
        
        if img_save_subfolder:

            save_dir = os.path.join(self.default_img_folder, img_save_subfolder)
            os.makedirs(save_dir, exist_ok=True)
        else:

            save_dir = self.default_img_folder
        if img_name is None:
            img_name = f"fig_{datetime.now().strftime('%H%M%S%f')}.png"
        img_path = os.path.join(save_dir, img_name)
        fig.savefig(img_path, bbox_inches='tight')
        
        if FIG_CLOSE:
            plt.close(fig)
        relative_path = os.path.relpath(img_path, self.log_folder)
        self._log_text(f"IMAGE_SAVED: {relative_path} | {description}", "INFO")
        return img_path
    
main_file_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
base_log_dir = os.path.join(main_file_dir, "logs")
logger = Logger(log_name="mc", base_log_dir=base_log_dir)

if __name__ == "__main__":
    logger = Logger(log_name="signal_processing", base_log_dir="logs")
    
    logger.log("### Signal processing experiment start ###", level="INFO")
    
    config = {
        "model": "CNN-LSTM", 
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32
    }
    logger.log(config, level="INFO")
    
    import numpy as np

    x = np.linspace(0, 4*np.pi, 200)
    origin_data = np.sin(x) * np.exp(-0.1*x)
    predict_data = origin_data + np.random.normal(0, 0.1, 200)
    points = [50, 100, 150]
    
    img_path = logger.log(
        origin_data,
        description="title",
        predict_signal=predict_data,
        # point_list=points,
    )
    
    
    # fig = plt.figure(figsize=(10, 4))
    # plt.legend()
    # plt.close(fig)

    
    
