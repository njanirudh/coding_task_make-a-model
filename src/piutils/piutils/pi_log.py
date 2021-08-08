#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is covered by the LICENSE file in the root of this project.
"""Same logging in the same format for all modules.
"""

import logging


def get_logger(
        name: str,
        logging_format: str = "[%(levelname)s %(name)s %(filename)s:%(funcName)s:%(lineno)d] %(message)s",
) -> logging.Logger:
    """Get a logger instance to be used in other modules.

    Usage:
        # at top of the module
        logger = get_logger(__name__)

        # later in the code
        logger.debug("This is a debug-level message.")
        logger.info("This is an info-level message.")
        logger.warning("This is a warning-level message.")
        logger.error("This is an error-level message.")

    Returns:
        A logging.Logger instance.
    """

    if name in logging.root.manager.loggerDict:
        # logger exists already
        return logging.root.manager.loggerDict[name]

    logger = logging.getLogger(name)

    if not logger.handlers:

        logging_stream_handler = logging.StreamHandler()

        if __debug__:
            logger.setLevel(logging.DEBUG)
            logging_stream_handler.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            logging_stream_handler.setLevel(logging.INFO)

        logging_formatter = logging.Formatter(
            logging_format, datefmt="%Y-%m-%dT%H:%M:%S"
        )
        logging_stream_handler.setFormatter(logging_formatter)

        logger.propagate = False

        logger.addHandler(logging_stream_handler)

    return logger
