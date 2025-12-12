# Copyright 2025 Optuna, HuggingFace Inc. and the LlamaFactory team. and Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Logging utils"""
# Based on: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/utils/logging.py

import logging
import os
import sys
import threading
from functools import lru_cache
from typing import Optional

import colorlog


_thread_lock = threading.RLock()
_is_configured = False  # Global flag to ensure setup is run only once


class _Logger(logging.Logger):
    """
    A logger that supports info_rank0.
    """

    def info_rank0(self, msg: str) -> None:
        if int(os.getenv("RANK", "0")) == 0:
            self.info(msg, stacklevel=2)

    def warning_rank0(self, msg: str) -> None:
        if int(os.getenv("RANK", "0")) == 0:
            self.warning(msg, stacklevel=2)

    @lru_cache(None)
    def warning_once(self, msg: str) -> None:
        if int(os.getenv("RANK", "0")) == 0:
            self.warning(msg, stacklevel=2)

    def debug_rank0(self, msg: str) -> None:
        if int(os.getenv("RANK", "0")) == 0:
            self.debug(msg, stacklevel=2)


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> "logging.Logger":
    logging.setLoggerClass(_Logger)
    return logging.getLogger(_get_library_name())


def add_file_handler(logging_dir: str) -> None:
    """
    Dynamically adds a file handler to the root logger on the main process.

    This function is idempotent; it will not add a new handler if one
    already exists. It can be called at any point after the initial setup.

    Args:
        logging_dir (str): The directory where the `train.log` file will be created.
    """
    with _thread_lock:
        rank = int(os.getenv("RANK", "0"))
        library_root_logger = _get_library_root_logger()

        # Only add the handler on the main process
        if rank != 0:
            return

        # Check if a file handler already exists to avoid duplicates
        if any(isinstance(h, logging.FileHandler) for h in library_root_logger.handlers):
            library_root_logger.debug("File handler already exists. Skipping.")
            return

        # Create and add the new file handler
        os.makedirs(logging_dir, exist_ok=True)
        log_file = os.path.join(logging_dir, "train.log")

        # Formatter for file (with colors)
        file_formatter = colorlog.ColoredFormatter(
            fmt="[%(log_color)s%(levelname)s%(reset)s][%(purple)s%(filename)s:%(lineno)s%(reset)s] %(cyan)s%(asctime)s%(reset)s >> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        library_root_logger.addHandler(file_handler)
        library_root_logger.info(f"Logging to file: {log_file}")


def setup_logging(
    level: str = "INFO",
    logging_dir: Optional[str] = None,
) -> None:
    """
    Configures the root logger for the library.

    This function is idempotent and thread-safe. It can be called multiple times,
    but the configuration will only be applied once. It sets up a colored
    console logger and an optional file logger (for rank 0 only).

    The logging level is determined in the following order of precedence:
    1. The `level` argument passed to this function.
    2. The `VEOMNI_VERBOSITY` environment variable.
    3. The default level, which is "INFO".

    Args:
        level (Optional[str]): The desired logging level (e.g., "DEBUG", "INFO", "WARNING").
        logging_dir (Optional[str]): If provided, logs from the main process (rank 0)
                                    will be written to a `train.log` file in this directory.
    """
    global _is_configured
    with _thread_lock:
        if not _is_configured:
            # Determine the logging level
            env_level = os.getenv("VEOMNI_VERBOSITY", "INFO").upper()
            log_level_str = (level or env_level).upper()

            try:
                log_level = logging.getLevelName(log_level_str)
            except ValueError:
                print(f"Warning: Invalid log level '{log_level_str}'. Defaulting to INFO.")
                log_level = logging.INFO

            # Get the root logger and configure it
            library_root_logger = _get_library_root_logger()
            library_root_logger.handlers.clear()
            library_root_logger.setLevel(log_level)
            library_root_logger.propagate = False

            # Formatter for console (with colors)
            console_formatter = colorlog.ColoredFormatter(
                fmt="[%(log_color)s%(levelname)s%(reset)s][%(purple)s%(filename)s:%(lineno)s%(reset)s] %(cyan)s%(asctime)s%(reset)s >> %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            # Console handler (for all ranks)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            library_root_logger.addHandler(console_handler)

            _is_configured = True

    if logging_dir:
        add_file_handler(logging_dir)


def get_logger(name: Optional[str] = None) -> _Logger:
    """
    Returns a logger with the specified name, ensuring the logging system is configured.

    This is the main entry point for obtaining a logger. It will lazily
    initialize the logging configuration on its first call. Any logger requested
    will be a child of the main library logger, inheriting its configuration.

    Args:
        name (Optional[str]): The name of the logger. If None, the root
                              library logger is returned.

    Returns:
        VeOmniLogger: An instance of the custom logger.
    """
    # Ensure the logging system is set up before returning a logger.
    setup_logging()

    root_logger_name = _get_library_name()
    if name:
        # Create a child logger of the root library logger
        logger_name = f"{root_logger_name}.{name}"
    else:
        logger_name = root_logger_name

    return logging.getLogger(logger_name)


def set_verbosity_info() -> None:
    """
    Sets the verbosity to the `INFO` level.
    """
    setup_logging()
    _get_library_root_logger().setLevel(logging.INFO)


def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("RANK", "0")) == 0:
        self.info(*args, **kwargs)


logging.Logger.info_rank0 = info_rank0


def debug_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("RANK", "0")) == 0:
        self.debug(*args, **kwargs)


logging.Logger.debug_rank0 = debug_rank0


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("RANK", "0")) == 0:
        self.warning(*args, **kwargs)


logging.Logger.warning_rank0 = warning_rank0


@lru_cache(None)
def warning_once(self, *args, **kwargs) -> None:
    if int(os.getenv("RANK", "0")) == 0:
        self.warning_rank0(*args, **kwargs)


logging.Logger.warning_once = warning_once