"""Shared imports and helpers for temporal scene graph modules."""

import json
import re
import time
from enum import Enum

from neo4j import GraphDatabase

from clivis import utils
from clivis.models.llm import *
from clivis.preference import NEO4J_AUTH, NEO4J_URI
from clivis.video import spilit_video

URI = NEO4J_URI
AUTH = NEO4J_AUTH


def remove_time_decimals(time_range):
    """
    Remove the decimal part from a time range.

    Args:
        time_range (str): A time range that may include decimals, e.g.
            "00:02:51.4-00:03:00.5" or a single timestamp like "00:02:51.4".

    Returns:
        str: A time range without decimals, e.g. "00:02:51-00:03:00" or
            a single timestamp like "00:02:51".
    """
    # If input is None or an empty string, return as-is.
    if not time_range:
        return time_range

    # Check whether input contains a '-' separator (time range).
    if '-' in time_range:
        start_time, end_time = time_range.split('-')
        # Handle start and end timestamps.
        start_time_no_decimal = remove_decimal_from_time(start_time)
        end_time_no_decimal = remove_decimal_from_time(end_time)
        # Compose the new time range.
        return f"{start_time_no_decimal}-{end_time_no_decimal}"
    else:
        # Handle a single timestamp.
        return remove_decimal_from_time(time_range)


def remove_decimal_from_time(time_str):
    """
    Remove the decimal part from a single timestamp.

    Args:
        time_str (str): A timestamp that may include decimals, e.g. "00:02:51.4".

    Returns:
        str: Timestamp without decimals, e.g. "00:02:51".
    """
    # If the timestamp includes a decimal part, keep the portion before '.'.
    if '.' in time_str:
        return time_str.split('.')[0]
    return time_str
