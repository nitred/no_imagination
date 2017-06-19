"""File containing basic utility functions."""

import datetime
import logging
import os

from PIL import Image

import numpy as np

logger = logging.getLogger(__name__)


def get_project_directory(*subdirs):
    """Create and return project home directory."""
    project_name = "no_imagination"
    project_dir = os.path.expanduser(os.path.join('~', '.{}'.format(str(project_name))))

    if subdirs:
        project_dir = os.path.join(project_dir, *subdirs)

    if not os.path.exists(project_dir):
        logger.debug("Creating project directory: {}".format(project_dir))
        os.makedirs(project_dir)

    return project_dir


def get_current_timestamp():
    """Return current timestamp in string format YYYY-MM-DD HH:MM:SS."""
    return datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
