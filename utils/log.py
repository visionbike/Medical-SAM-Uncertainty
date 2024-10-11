from typing import Dict
import time
import logging
from logging import Logger
from datetime import datetime
from dateutil.tz import tzlocal
from pathlib import Path

__all__ = [
    "create_log_directory",
    "create_logger"
]

def create_log_directory(root: str, exp_name: str) -> Dict[str, str]:
    """
    Create logging directory.

    Args:
        root (str): logging root directory.
        exp_name: defined experimental name.
    Returns:
        (Dict): created paths for logging.
    """
    path_root = Path(root)
    path_root.mkdir(parents=True, exist_ok=True)

    # create experimental path
    now = datetime.now(tzlocal())
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    path_exp = path_root / f"{exp_name}_{timestamp}"
    path_exp.mkdir(parents=True, exist_ok=True)

    # create checkpoint path
    path_ckpt = path_exp / "ckpt"
    path_ckpt.mkdir(parents=True, exist_ok=True)

    # create log path
    path_log = path_exp / "log"
    path_log.mkdir(parents=True, exist_ok=True)

    # create sample path for image and visualization
    path_sample = path_exp / "sample"
    path_sample.mkdir(parents=True, exist_ok=True)

    # create tensorboard path
    path_run = path_exp / "run" / timestamp
    path_run.mkdir(parents=True, exist_ok=True)

    return {
        "path_exp": path_exp.__str__(),
        "path_ckpt": path_ckpt.__str__(),
        "path_log": path_log.__str__(),
        "path_sample": path_sample.__str__(),
        "path_run": path_run.__str__()
    }


def create_logger(log_dir: str, phase="train") -> Logger:
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f"{time_str}_{phase}.log"
    final_log_file = Path(log_dir) / log_file
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)
    return logger
