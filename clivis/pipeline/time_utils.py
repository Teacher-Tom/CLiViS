"""Time and question parsing helpers for inference pipelines."""

import re

from clivis.video import spilit_video


DEFAULT_MAX_FPS = 8


def extract_question(text):
    """Extract the question part from a prompt that may also contain choices."""
    pattern = r"question:\s+(.*?)(?=\s*(?:choices:|choice:|option \d+:|[A-D]\.)|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def calculate_video_segments(
    video_duration,
    max_segments=10,
    base_unit=15,
    min_segment_length=30,
):
    """Split a video duration into coarse hh:mm:ss-hh:mm:ss ranges."""
    if video_duration <= 15:
        return [f"00:00:00-{spilit_video.seconds_to_time_str(video_duration)}"]

    ideal_segment_length = video_duration / max_segments
    if ideal_segment_length < min_segment_length:
        segment_length = min_segment_length
    else:
        multiplier = max(1, int((ideal_segment_length + base_unit - 1) // base_unit))
        segment_length = multiplier * base_unit

    segment_count = min(
        max_segments,
        int(video_duration // segment_length)
        + (1 if video_duration % segment_length > 0 else 0),
    )
    last_segment_length = video_duration - (segment_count - 1) * segment_length
    if segment_count > 1 and last_segment_length <= 0.2 * segment_length:
        segment_count -= 1

    period_names = []
    for i in range(segment_count):
        start_time = i * segment_length
        end_time = video_duration if i == segment_count - 1 else (i + 1) * segment_length
        period_names.append(
            f"{spilit_video.seconds_to_time_str(start_time)}-"
            f"{spilit_video.seconds_to_time_str(end_time)}"
        )
    return period_names


def calculate_period_fps(period, fps, video_duration, max_fps=DEFAULT_MAX_FPS):
    """Choose a bounded FPS for querying a specific period."""
    try:
        period_start, period_end = period.split("-")
        period_start = spilit_video.time_str_to_seconds(period_start)
        period_end = spilit_video.time_str_to_seconds(period_end)
        period_length = period_end - period_start
        if period_length <= 0:
            return fps
        return min(min(4, video_duration // period_length) * fps, max_fps)
    except Exception as exc:
        print(f"Error calculating period fps: {exc}")
        return fps


def is_valid_period_format(period):
    """Return whether period is ``full video`` or ``hh:mm:ss-hh:mm:ss``."""
    if period == "full video":
        return True
    if not isinstance(period, str) or "-" not in period:
        return False

    parts = period.split("-")
    if len(parts) != 2:
        return False

    for time_str in parts:
        time_parts = time_str.strip().split(":")
        if len(time_parts) != 3:
            return False
        try:
            hh, mm, ss = (int(part) for part in time_parts)
        except ValueError:
            return False
        if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
            return False
    return True
