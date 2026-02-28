import logging
import tempfile
import asyncio
import uvicorn
import torch
import gc
from concurrent.futures import ThreadPoolExecutor
from transcription import Transcriber
from pyannote.audio import Pipeline
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware

from utils import assign_words_to_speakers, build_utterances_from_words   

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Thread pool for CPU-bound tasks (ASR + Diarization)
executor = ThreadPoolExecutor(max_workers=4)

# ─── Concurrency control ────────────────────────────────────────────────────
# Limit concurrent GPU-heavy requests to prevent OOM
# Adjust max value based on your GPU memory:
#   - 46GB VRAM: 2-3 concurrent long audio is safe
#   - 24GB VRAM: 1 concurrent long audio
MAX_CONCURRENT_TRANSCRIPTIONS = 2
_transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)

# ─── Lazy singletons ────────────────────────────────────────────────────────

transcriber = None
diarization_pipeline = None

def get_transcriber():
    global transcriber
    if transcriber is None:
        logger.info("Initializing Transcriber...")
        transcriber = Transcriber(model_path="ckpt/asr/")
    return transcriber

def get_diarization_pipeline():
    global diarization_pipeline
    if diarization_pipeline is None:
        logger.info("Initializing Diarization Pipeline...")
        diarization_pipeline = Pipeline.from_pretrained(
            "ckpt/segment/",
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diarization_pipeline.to(device)
        logger.info(f"Diarization pipeline loaded on {device}")
    return diarization_pipeline

# ─── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Loading models...")
    loop = asyncio.get_event_loop()
    await asyncio.gather(
        loop.run_in_executor(executor, get_transcriber),
        loop.run_in_executor(executor, get_diarization_pipeline),
    )
    logger.info("All models loaded successfully")

# ─── Middleware ──────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _run_asr(audio_bytes: bytes, chunk_size, left_context_size,
            right_context_size, total_batch_duration, 
            max_silence_duration, 
            return_word_timestamps=False):
    """Run ASR synchronously – called inside thread-pool executor."""
    trans = get_transcriber()
    segments, full_transcript = trans.transcribe_audio(
        audio_path=audio_bytes,
        chunk_size=chunk_size,
        left_context_size=left_context_size,
        right_context_size=right_context_size,
        total_batch_duration=total_batch_duration,
        max_silence_duration=max_silence_duration,
        return_word_timestamps=return_word_timestamps,
    )
    # Force garbage collection after ASR to free any lingering tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return segments, full_transcript

def _run_diarization(tmp_path: str):
    """Run diarization synchronously – called inside thread-pool executor."""
    pipeline = get_diarization_pipeline()
    output = pipeline(tmp_path)
    speaker_turns = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, speaker in output.speaker_diarization
    ]
    # Force cleanup after diarization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return speaker_turns

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"This is": "STT API (Transcriber)"}


@app.post("/speech-to-text")
async def transcribe_audio(
    file: UploadFile = File(...),
    chunk_size: int = Query(64,   description="Chunk size for processing"),
    left_context_size:  int = Query(128,  description="Left context size"),
    right_context_size: int = Query(128,  description="Right context size"),
    total_batch_duration: int = Query(1800, description="Total batch duration in seconds"),
    max_silence_duration: float = Query(0.5, description="Maximum silence duration for sentence breaks"),
):
    """Standard speech-to-text without speaker labels."""
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"error": "Empty audio file", "transcriptions": []}

    async with _transcription_semaphore:
        try:
            trans = get_transcriber()
            loop = asyncio.get_event_loop()
            segments, full_transcript = await loop.run_in_executor(
                executor,
                _run_asr,
                audio_bytes, chunk_size, left_context_size,
                right_context_size, total_batch_duration, max_silence_duration,
                False,
            )
            logger.info(f"Transcription result: {full_transcript}")
            return {"transcriptions": full_transcript, "details": segments}
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}", exc_info=True)
            return {"error": str(e), "transcriptions": []}


@app.post("/meeting-transcribe")
async def meeting_transcribe(
    file: UploadFile = File(...),
    chunk_size: int = Query(64,   description="Chunk size for ASR processing"),
    left_context_size:  int = Query(128,  description="Left context size"),
    right_context_size: int = Query(128,  description="Right context size"),
    total_batch_duration: int = Query(1800, description="Total batch duration in seconds"),
    max_silence_duration: float = Query(0.5, description="Maximum silence duration"),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"error": "Empty audio file", "transcript": []}

    # Write to temp file (pyannote requires a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Acquire semaphore to limit concurrent GPU usage
        async with _transcription_semaphore:
            loop = asyncio.get_event_loop()

            # ── Run ASR and Diarization in PARALLEL ──────────────────────
            asr_future = loop.run_in_executor(
                executor,
                _run_asr,
                audio_bytes, chunk_size, left_context_size,
                right_context_size, total_batch_duration, max_silence_duration,
                True,  # return_word_timestamps=True 
            )
            diarization_future = loop.run_in_executor(
                executor,
                _run_diarization,
                tmp_path,
            )

            (asr_segments, _), speaker_turns = await asyncio.gather(
                asr_future, diarization_future
            )

        # ── Post-processing runs on CPU only, no semaphore needed ────────
        logger.info(f"ASR segments: {len(asr_segments)} | Speaker turns: {len(speaker_turns)}")

        # Collect all words from all ASR segments
        all_words = []
        for seg in asr_segments:
            if "words" in seg:
                all_words.extend(seg["words"])

        # Assign each word to a speaker using binary search
        labeled_words = assign_words_to_speakers(all_words, speaker_turns)

        # Merge consecutive same-speaker words → utterances
        utterances = build_utterances_from_words(labeled_words)

        logger.info(f"Built {len(utterances)} utterances")

        return {
            "utterances": utterances,
            "speaker_turns": speaker_turns,
        }

    except Exception as e:
        logger.error(f"Meeting transcription error: {str(e)}", exc_info=True)
        return {"error": str(e), "transcript": []}

    finally:
        import os
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)