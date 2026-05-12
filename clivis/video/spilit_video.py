from moviepy.video.io.VideoFileClip import VideoFileClip
import re
import os

import cv2


def round_to_xx_seconds(seconds, round_to=15, mode="floor"):
    """Round to 15-second units by default, rounding down."""
    if mode == "floor":
        return (seconds // round_to) * round_to
    elif mode == "ceil":
        return ((seconds + round_to - 1) // round_to) * round_to  # Round up.
    else:
        return seconds

def adjust_time_segment(start, end, video_duration):
    """Round a time segment to 15-second units and keep it within video bounds."""
    # Round the start time down.
    adjusted_start = round_to_xx_seconds(start, round_to=1, mode="floor")
    # Round the end time up.
    adjusted_end = round_to_xx_seconds(end, round_to=1, mode="ceil")

    # Clamp the adjusted times to the video duration.
    adjusted_start = max(0, adjusted_start)
    adjusted_end = min(video_duration, adjusted_end)

    return adjusted_start, adjusted_end


def get_video_duration(video_path):
    """
    Get the video duration in seconds.
    :param video_path: Path to the video file
    :return: Video duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    # Get the total frame count and FPS.
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Compute the video duration in seconds.
    duration = frame_count / fps

    cap.release()
    return duration


def time_str_to_seconds(time_str):
    """Convert a time string (hh:mm:ss) to seconds."""
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 3:  # hh:mm:ss format
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:  # Support legacy mm:ss format.
        minutes, seconds = parts
        return minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}")

def seconds_to_time_str(seconds):
    """Convert seconds to a time string in hh:mm:ss format."""
    # Ensure integer arithmetic.
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def find_time_range(query_time_range, time_ranges):
    """
    Find time ranges that overlap with the query range, e.g. "00:00:12-00:00:13".
    :param query_time_range: Query time range string
    :param time_ranges: List of time ranges in ["hh:mm:ss-hh:mm:ss", ...] format
    """
    start_str, end_str = query_time_range.split('-')
    start_sec = time_str_to_seconds(start_str)
    end_sec = time_str_to_seconds(end_str)
    result = []
    for time_range in time_ranges:
        time_start_str, time_end_str = time_range.split('-')
        time_start_sec = time_str_to_seconds(time_start_str)
        time_end_sec = time_str_to_seconds(time_end_str)

        # Check whether the time ranges overlap.
        if (start_sec < time_end_sec) and (end_sec > time_start_sec):
            result.append(time_range)
    return result



def split_video(input_path, segments, output_prefix="output/output", target_resolution=(360, 640)):
    """
    Split a video file using moviepy instead of ffmpeg.

    Args:
        input_path: Input video file path
        segments: List of time segments in ["hh:mm:ss-hh:mm:ss", ...] format
        output_prefix: Output file prefix

    Returns:
        segments_to_files: Dictionary mapping original time segments to output file paths
    """
    segments_to_files = {}

    try:
        # Ensure the output directory exists.
        output_dir = os.path.dirname(output_prefix)
        if output_dir:
            # If the directory already exists, clear its files.
            if os.path.exists(output_dir):
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print(f"Cleared output directory: {output_dir}")
            else:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")


        # Load the video.
        video = VideoFileClip(input_path, audio=False)
        video_duration = video.duration

        for i, segment in enumerate(segments, 1):
            # Parse the time segment (hh:mm:ss format supported).
            match = re.search(r"^(\d{2}:\d{2}:\d{2})-(\d{2}:\d{2}:\d{2})$", segment)
            if not match:
                print(f"Skipping invalid time segment format: {segment}")
                continue

            start_str, end_str = match.groups()
            start_sec = time_str_to_seconds(start_str)
            end_sec = time_str_to_seconds(end_str)

            # Check whether the time range is valid.
            if start_sec >= end_sec:
                print(f"Time segment {segment} has start >= end; skipping.")
                continue

            # Round to 15-second units and keep within the video duration.
            adjusted_start, adjusted_end = adjust_time_segment(start_sec, end_sec, video_duration)

            # If the adjusted range is invalid, skip it.
            if adjusted_start >= adjusted_end:
                print(
                    f"Adjusted segment {seconds_to_time_str(adjusted_start)}-{seconds_to_time_str(adjusted_end)} is invalid; skipping.")
                continue

            # Trim the video clip.
            clip = video.subclipped(adjusted_start, adjusted_end)

            # Build the output file name.
            output_path = f"{output_prefix}_{i:03d}.mp4"

            # Write the clip to disk.
            clip.write_videofile(output_path, codec="libx264", audio=False, preset="ultrafast")
            print(
                f"Saved clip {i}: {seconds_to_time_str(adjusted_start)}-{seconds_to_time_str(adjusted_end)} -> {output_path}")

            # Add the original time segment to the mapping.
            segments_to_files[segment] = output_path

        print("Video splitting complete!")
        return segments_to_files
    except Exception as e:
        print(f"Error while processing video: {e}")
        return segments_to_files
    finally:
        if 'video' in locals():
            video.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split a video into time segments.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--segment", action="append", required=True, help="Time segment, e.g. 00:00:00-00:00:15")
    parser.add_argument("--output-prefix", default="segments/output", help="Output file prefix")
    args = parser.parse_args()

    time_segments = [
        *args.segment,
    ]

    split_video(args.video, time_segments, output_prefix=args.output_prefix)
