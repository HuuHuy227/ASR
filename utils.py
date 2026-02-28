"""
Utility functions for the STT api
"""

import torch 
import logging
import mimetypes
import bisect
from typing import Dict, List

logger = logging.getLogger(__name__)

def get_gpu_capability():
    if not torch.cuda.is_available():
        return "❌ No GPU available"
    
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    major, minor = capability
    if major >= 8:
        return "bf16"
    elif major == 7:
        return "fp16"
    elif major >= 6:
        return "partial_fp16"
    else:
        return "❌ No native FP16/BF16 support"

def timestamp_to_seconds(timestamp_str: str) -> float:
    """Convert hh:mm:ss:ms format to seconds

    Args:
        timestamp_str: Timestamp in format '00:00:05:123'

    Returns:
        Total seconds as float (e.g., 5.123)
    """
    try:
        parts = str(timestamp_str).split(":")
        if len(parts) == 4:
            hours, minutes, seconds, milliseconds = map(int, parts)
            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
            return total_seconds
        else:
            # Try parsing as float directly
            return float(timestamp_str)
    except (ValueError, AttributeError, TypeError):
        logger.warning(f"Could not parse timestamp: {timestamp_str}, using 0.0")
        return 0.0


def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS format

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string in MM:SS format
    """
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"

def create_subtitle_srt(segments: List[Dict]) -> str:
    """Create SRT format subtitles

    Args:
        segments: List of segment dictionaries with start, end, and text

    Returns:
        SRT formatted subtitle string
    """
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        srt_content += f"{i}\n{start} --> {end}\n{segment['text']}\n\n"
    return srt_content


def guess_video_mime_type(file_name: str) -> str:
    """Best-effort MIME type detection for video assets

    Args:
        file_name: Name of the video file

    Returns:
        MIME type string (defaults to video/mp4)
    """
    if not file_name:
        return "video/mp4"
    mime_type, _ = mimetypes.guess_type(file_name)
    if mime_type and mime_type.startswith("video/"):
        return mime_type
    return "video/mp4"


def get_transcript_at_time(
    segments: List[Dict], current_time: float, context_window: float = 5.0
) -> str:
    """Get transcript text for current playback time with context

    Args:
        segments: List of transcript segments
        current_time: Current playback time in seconds
        context_window: Time window in seconds for context

    Returns:
        Formatted transcript text with current segment highlighted
    """
    transcript_lines = []

    for segment in segments:
        if segment["start"] - context_window <= current_time <= segment["end"] + context_window:
            # Highlight current segment
            if segment["start"] <= current_time <= segment["end"]:
                transcript_lines.append(f"► {segment['text']} ◄")
            else:
                transcript_lines.append(segment["text"])

    return " ".join(transcript_lines) if transcript_lines else "Loading transcript..."


def prepare_segments_for_player(segments: List[Dict]) -> List[Dict]:
    """Normalize segment payload for the synchronized player

    Args:
        segments: Raw segment data from transcription

    Returns:
        Normalized segments with index, start, end, and text
    """
    prepared_segments: List[Dict] = []
    for idx, segment in enumerate(segments, start=1):
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        if end <= start:
            end = start + 0.01  # ensure strictly increasing to avoid zero-length highlights
        prepared_segments.append(
            {
                "index": idx,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": segment.get("text", ""),
            }
        )
    return prepared_segments

def assign_words_to_speakers(words: list, speaker_turns: list) -> list:
    """
    Assign each word to a speaker based on word midpoint vs diarization turns.

    Optimized with binary search: O(W × log T) instead of O(W × T).

    Args:
        words: [{"text", "start_ms", "end_ms"}, ...]
        speaker_turns: [{"start": float_seconds, "end": float_seconds, "speaker": str}, ...]

    Returns:
        Same word list with "speaker" key added to each word.
    """
    if not speaker_turns or not words:
        return words

    # Pre-sort turns by start time and build index arrays for binary search
    sorted_turns = sorted(speaker_turns, key=lambda t: t["start"])
    turn_starts_ms = [int(t["start"] * 1000) for t in sorted_turns]
    turn_ends_ms   = [int(t["end"]   * 1000) for t in sorted_turns]
    turn_speakers  = [t["speaker"] for t in sorted_turns]
    n_turns = len(sorted_turns)

    for word in words:
        word_mid = (word["start_ms"] + word["end_ms"]) / 2.0
        word_mid_int = int(word_mid)

        # Binary search: find the rightmost turn whose start_ms <= word_mid
        # This is the most likely candidate for midpoint containment
        idx = bisect.bisect_right(turn_starts_ms, word_mid_int) - 1

        best_speaker = None

        # Priority 1: Check if midpoint falls inside the turn at idx
        if 0 <= idx < n_turns and turn_starts_ms[idx] <= word_mid_int <= turn_ends_ms[idx]:
            best_speaker = turn_speakers[idx]
        else:
            # Priority 2: Check neighboring turns for overlap
            # Only need to check a small window around idx
            best_overlap = 0
            for check_idx in range(max(0, idx - 1), min(n_turns, idx + 3)):
                overlap_start = max(word["start_ms"], turn_starts_ms[check_idx])
                overlap_end   = min(word["end_ms"],   turn_ends_ms[check_idx])
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn_speakers[check_idx]

            # Priority 3: Nearest turn fallback
            if best_speaker is None:
                nearest_dist = float("inf")
                # Check idx and idx+1 (the two closest turns by start time)
                for check_idx in range(max(0, idx), min(n_turns, idx + 2)):
                    dist = min(
                        abs(word_mid_int - turn_starts_ms[check_idx]),
                        abs(word_mid_int - turn_ends_ms[check_idx]),
                    )
                    if dist < nearest_dist:
                        nearest_dist = dist
                        best_speaker = turn_speakers[check_idx]
                # Also check idx-1 if it exists
                if idx - 1 >= 0:
                    dist = abs(word_mid_int - turn_ends_ms[idx - 1])
                    if dist < nearest_dist:
                        best_speaker = turn_speakers[idx - 1]

        word["speaker"] = best_speaker or "UNKNOWN"

    return words


def build_utterances_from_words(words: list) -> list:
    """Merge consecutive same-speaker words into utterances."""
    if not words:
        return []

    utterances = []
    current_speaker = words[0].get("speaker", "UNKNOWN")
    current_words = [words[0]]

    for word in words[1:]:
        speaker = word.get("speaker", "UNKNOWN")
        if speaker == current_speaker:
            current_words.append(word)
        else:
            utterances.append({
                "speaker": current_speaker,
                "start": format_timestamp(current_words[0]["start_ms"] / 1000.0),
                "end": format_timestamp(current_words[-1]["end_ms"] / 1000.0),
                "text": " ".join(w["text"] for w in current_words),
            })
            current_speaker = speaker
            current_words = [word]

    utterances.append({
        "speaker": current_speaker,
        "start": format_timestamp(current_words[0]["start_ms"] / 1000.0),
        "end": format_timestamp(current_words[-1]["end_ms"] / 1000.0),
        "text": " ".join(w["text"] for w in current_words),
    })

    return utterances