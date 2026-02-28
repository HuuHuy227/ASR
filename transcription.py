"""
Transcription module for Chunkformer model
"""

import io
import time
import torch
import logging
from typing import Dict, List, Tuple
from utils import timestamp_to_seconds, get_gpu_capability
from contextlib import nullcontext, redirect_stderr, redirect_stdout

logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self, model_path : str = "chunkformer-ctc-large-vie"):
        gpu_type = get_gpu_capability()
        autocast_type = gpu_type if gpu_type=='bf16' else None
        self.dtype = {"bf16": torch.bfloat16, None: None}[autocast_type]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self.load_model()
        
    def load_model(self):
        """Load and cache the Chunkformer model

        Args:
            model_path: Path or HuggingFace ID of the model

        Returns:
            Loaded ChunkFormer model or None if failed
        """
        try:
            # Import here to avoid circular dependencies
            from chunkformer import ChunkFormerModel

            logger.info(f"[LOAD_MODEL] Starting to load model: {self.model_path}")
            start_time = time.time()

            model = ChunkFormerModel.from_pretrained(self.model_path)
            model = model.to(self.device)
            model.eval()

            elapsed = time.time() - start_time
            logger.info(f"[LOAD_MODEL] Model loaded successfully in {elapsed:.2f}s")
            return model
        except Exception as e:
            logger.error(f"[LOAD_MODEL] Failed to load model: {e}", exc_info=True)
            return None

    def transcribe_audio(
        self,
        audio_path,
        chunk_size: int = 64,
        left_context_size: int = 128,
        right_context_size: int = 128,
        total_batch_duration: int = 1800,
        max_silence_duration: float = 0.5,
        return_word_timestamps: bool = False,
    ) -> Tuple[List[Dict], str]:
        """Transcribe audio/video using Chunkformer model's endless_decode

        Accepts both audio files (.wav, .mp3, etc.) and video files (.mp4, .mkv, etc.)

        Args:
            model: Loaded Chunkformer model
            media_path: Path to audio or video file
            chunk_size: Size of chunks for processing
            left_context_size: Left context window size
            right_context_size: Right context window size
            total_batch_duration: Total batch duration in seconds
            max_silence_duration: Maximum silence duration in seconds for sentence break detection

        Returns:
            Tuple of (segments list, full transcript string)

        The endless_decode method returns segments with timing information in format:
        [{'start': time_in_seconds, 'end': time_in_seconds, 'decode': text}, ...]
        """
        try:
                overall_start = time.time()
                decode_start = time.time()

                # Suppress stderr/stdout to avoid tqdm broken pipe errors
                with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()), \
                (torch.autocast(self.device.type, self.dtype) if self.dtype is not None else nullcontext()):
                    decode_result = self.model.endless_decode(
                        audio_path=audio_path,  # Can be audio 
                        chunk_size=chunk_size,
                        left_context_size=left_context_size,
                        right_context_size=right_context_size,
                        total_batch_duration=total_batch_duration,
                        return_timestamps=True,
                        max_silence_duration=max_silence_duration,
                        return_word_timestamps=return_word_timestamps,
                    )

                decode_elapsed = time.time() - decode_start
                logger.info(f"[TRANSCRIBE] endless_decode completed in {decode_elapsed:.2f}s")

                # Convert decode_result to segments format
                segments = []
                full_transcript = ""

                if isinstance(decode_result, list):
                    for item in decode_result:
                        if isinstance(item, dict):
                            # Item format:
                            # {'start': 'hh:mm:ss:ms', 'end': 'hh:mm:ss:ms',
                            #  'decode': str}
                            start_str = str(item.get("start", "00:00:00:000"))
                            end_str = str(item.get("end", "00:00:00:000"))

                            start_float = timestamp_to_seconds(start_str)
                            end_float = timestamp_to_seconds(end_str)

                            segment = {
                                "start": start_float,
                                "end": end_float,
                                "text": item.get("text", ""),
                            }
                            
                            # Include word-level timestamps if available
                            if return_word_timestamps and "words" in item:
                                segment["words"] = item["words"]

                            segments.append(segment)
                            full_transcript += item.get("text", "") + " "

                overall_elapsed = time.time() - overall_start
                logger.info(
                    f"[TRANSCRIBE] Transcription complete! "
                    f"{len(segments)} segments in {overall_elapsed:.2f}s"
                )

                return segments, full_transcript.strip()

        except Exception as e:
            logger.error(f"[TRANSCRIBE] Error during transcription: {e}", exc_info=True)
            return [], ""
