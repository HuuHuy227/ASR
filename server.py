import io
import logging
import asyncio
import uvicorn
import torch
import gc
import torchaudio
from concurrent.futures import ThreadPoolExecutor
from transcription import Transcriber
from pyannote.audio import Pipeline
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import json
from utils import assign_words_to_speakers, build_utterances_from_words

# Tăng tốc tính toán ma trận trên Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Thread pool cho CPU-bound / blocking tasks
executor = ThreadPoolExecutor(max_workers=4)

# ─── Concurrency control ─────────────────────────────────────────────────────
# Giới hạn số request dùng GPU cùng lúc để tránh OOM.
# Mỗi slot semaphore chỉ bọc MỘT bước GPU tại một thời điểm
# (ASR hoặc Diarization), KHÔNG bọc cả pipeline → tận dụng tối đa throughput.
MAX_CONCURRENT_GPU_TASKS = 2
_gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GPU_TASKS)

# ─── Lazy singletons ─────────────────────────────────────────────────────────
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
        diarization_pipeline = Pipeline.from_pretrained("ckpt/segment/")

        # Tăng batch size để GPU chạy hết công suất khi embedding
        diarization_pipeline.embedding_batch_size = 64
        diarization_pipeline.segmentation_batch_size = 64

        # Giảm over-segmentation: merge speaker turn ngắn hơn 0.5s
        # 2268 turns → ~400-600 turns sau khi merge
        diarization_pipeline.segmentation.min_duration_off = 0.5

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diarization_pipeline.to(device)
        logger.info(f"Diarization pipeline loaded on {device}")
    return diarization_pipeline


# ─── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Loading models...")
    loop = asyncio.get_event_loop()
    await asyncio.gather(
        loop.run_in_executor(executor, get_transcriber),
        loop.run_in_executor(executor, get_diarization_pipeline),
    )
    logger.info("All models loaded successfully")


# ─── Middleware ───────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ─── GPU Task Helpers ─────────────────────────────────────────────────────────

def _run_asr(
    audio_bytes: bytes,
    chunk_size: int,
    left_context_size: int,
    right_context_size: int,
    total_batch_duration: int,
    max_silence_duration: float,
    return_word_timestamps: bool = False,
):
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
    # Giải phóng VRAM trước khi trả về để bước tiếp theo có đủ memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return segments, full_transcript


def _run_diarization(audio_bytes: bytes):
    """
    Run diarization synchronously – called inside thread-pool executor.

    Nhận thẳng audio_bytes từ RAM, tránh hoàn toàn Disk I/O đọc file.
    torchaudio.load() với BytesIO nhanh hơn đọc từ temp file trên ổ cứng.
    """
    import time

    t0 = time.time()

    # Load audio thẳng từ RAM, không cần temp file
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
    logger.info(f"[DIARIZATION] Audio loaded to RAM: {waveform.shape}, sr={sample_rate}")

    pipeline = get_diarization_pipeline()
    output = pipeline(audio_dict)

    elapsed = time.time() - t0
    logger.info(f"[DIARIZATION] Pipeline done in {elapsed:.2f}s")

    speaker_turns = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, speaker in output.speaker_diarization
    ]

    logger.info(f"[DIARIZATION] Extracted {len(speaker_turns)} speaker turns")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return speaker_turns


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"This is": "STT API (Transcriber)"}


@app.post("/speech-to-text")
async def transcribe_audio(
    file: UploadFile = File(...),
    chunk_size: int = Query(64, description="Chunk size for processing"),
    left_context_size: int = Query(128, description="Left context size"),
    right_context_size: int = Query(128, description="Right context size"),
    total_batch_duration: int = Query(1800, description="Total batch duration in seconds"),
    max_silence_duration: float = Query(0.5, description="Maximum silence duration for sentence breaks"),
):
    """Standard speech-to-text without speaker labels."""
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"error": "Empty audio file", "transcriptions": []}

    try:
        loop = asyncio.get_event_loop()
        async with _gpu_semaphore:
            segments, full_transcript = await loop.run_in_executor(
                executor,
                _run_asr,
                audio_bytes, chunk_size, left_context_size,
                right_context_size, total_batch_duration, max_silence_duration,
                False,
            )
        logger.info(f"Transcription result: {full_transcript[:100]}...")
        return {"transcriptions": full_transcript, "details": segments}
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        return {"error": str(e), "transcriptions": []}

@app.post("/meeting-transcribe")
async def meeting_transcribe(
    file: UploadFile = File(...),
    chunk_size: int = Query(64, description="Chunk size for ASR processing"),
    left_context_size: int = Query(128, description="Left context size"),
    right_context_size: int = Query(128, description="Right context size"),
    total_batch_duration: int = Query(1800, description="Total batch duration in seconds"),
    max_silence_duration: float = Query(0.5, description="Maximum silence duration"),
):
    """
    Meeting transcription với speaker diarization.

    Luồng xử lý:
        [ASR]  → acquire GPU semaphore → chạy → release
                                                    ↓
        [DIAR] → acquire GPU semaphore → chạy → release
                                                    ↓
        [CPU]  → assign_words_to_speakers + build_utterances (không cần semaphore)

    Mỗi bước chỉ giữ semaphore trong thời gian nó thực sự dùng GPU.
    Khi ASR xong và đang chờ DIAR acquire semaphore, slot đó có thể
    được request khác dùng (ví dụ: /speech-to-text từ user khác).
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"error": "Empty audio file", "transcript": []}

    try:
        loop = asyncio.get_event_loop()

        # ── BƯỚC 1: ASR (GPU) ─────────────────────────────────────────────
        logger.info("[MEETING] Bắt đầu ASR...")
        async with _gpu_semaphore:
            asr_segments, _ = await loop.run_in_executor(
                executor, _run_asr,
                audio_bytes, chunk_size, left_context_size,
                right_context_size, total_batch_duration, max_silence_duration,
                True,  # return_word_timestamps=True
            )
        # _run_asr đã gọi gc.collect() + empty_cache() bên trong
        logger.info(f"[MEETING] ASR xong: {len(asr_segments)} segments")

        # ── BƯỚC 2: Diarization (GPU) ─────────────────────────────────────
        # Acquire semaphore lần hai — độc lập với bước 1.
        # Trong khoảng thời gian giữa release bước 1 và acquire bước 2,
        # request khác có thể dùng GPU slot đó.
        logger.info("[MEETING] Bắt đầu Diarization...")
        async with _gpu_semaphore:
            speaker_turns = await loop.run_in_executor(
                executor, _run_diarization,
                audio_bytes,  # truyền thẳng bytes, không cần temp file
            )
        logger.info(f"[MEETING] Diarization xong: {len(speaker_turns)} turns")

        # ── BƯỚC 3: Post-processing (CPU only, không cần semaphore) ───────
        all_words = []
        for seg in asr_segments:
            if "words" in seg:
                all_words.extend(seg["words"])

        logger.info(f"[MEETING] Post-processing {len(all_words)} words...")
        labeled_words = assign_words_to_speakers(all_words, speaker_turns)
        utterances = build_utterances_from_words(labeled_words)

        logger.info(f"[MEETING] Built {len(utterances)} utterances")

        return {
            "utterances": utterances,
            "speaker_turns": speaker_turns,
        }

    except Exception as e:
        logger.error(f"Meeting transcription error: {str(e)}", exc_info=True)
        return {"error": str(e), "transcript": []}

@app.post("/stream-meeting-transcribe")
async def stream_transcribe(
    file: UploadFile = File(...),
    chunk_size: int = Query(64),
    left_context_size: int = Query(128),
    right_context_size: int = Query(128),
    total_batch_duration: int = Query(1800),
    max_silence_duration: float = Query(0.5),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        async def empty_gen():
            yield json.dumps({"status": "failed", "error": "Empty audio file"}) + "\n"
        return StreamingResponse(empty_gen(), media_type="application/x-ndjson")

    async def process_generator():
        try:
            loop = asyncio.get_running_loop()  # get_event_loop() deprecated Python 3.10+

            # ── BƯỚC 1: ASR ──────────────────────────────────────────────
            yield json.dumps({"status": "processing", "step": "ASR", "percent": 5}) + "\n"
            logger.info("[STREAM] Bắt đầu ASR...")

            async with _gpu_semaphore:
                # FIX 1: truyền đủ tham số + unpack tuple (segments, full_transcript)
                asr_segments, _ = await loop.run_in_executor(
                    executor, _run_asr,
                    audio_bytes, chunk_size, left_context_size,
                    right_context_size, total_batch_duration, max_silence_duration,
                    True,  # return_word_timestamps=True
                )
            logger.info(f"[STREAM] ASR xong: {len(asr_segments)} segments")
            yield json.dumps({"status": "processing", "step": "ASR", "percent": 45}) + "\n"

            # ── BƯỚC 2: Diarization ───────────────────────────────────────
            yield json.dumps({"status": "processing", "step": "Diarization", "percent": 50}) + "\n"
            logger.info("[STREAM] Bắt đầu Diarization...")

            async with _gpu_semaphore:
                speaker_turns = await loop.run_in_executor(
                    executor, _run_diarization,
                    audio_bytes,
                )
            logger.info(f"[STREAM] Diarization xong: {len(speaker_turns)} turns")
            yield json.dumps({"status": "processing", "step": "Diarization", "percent": 88}) + "\n"

            # ── BƯỚC 3: Post-processing ───────────────────────────────────
            yield json.dumps({"status": "processing", "step": "Post-processing", "percent": 90}) + "\n"

            # FIX 2: extract all_words từ asr_segments trước khi dùng
            all_words = []
            for seg in asr_segments:
                if "words" in seg:
                    all_words.extend(seg["words"])
            logger.info(f"[STREAM] Post-processing {len(all_words)} words...")

            labeled_words = assign_words_to_speakers(all_words, speaker_turns)
            utterances = build_utterances_from_words(labeled_words)
            logger.info(f"[STREAM] Built {len(utterances)} utterances")

            yield json.dumps({
                "status": "completed",
                "percent": 100,
                "result": {
                    "utterances": utterances,
                    "speaker_turns": speaker_turns,
                }
            }) + "\n"

        except Exception as e:
            logger.error(f"[STREAM] Error: {e}", exc_info=True)
            yield json.dumps({"status": "failed", "error": str(e)}) + "\n"

    # FIX 3: thêm headers chống buffer proxy/nginx/uvicorn
    headers = {
        "X-Accel-Buffering": "no",   # tắt nginx buffering
        "Cache-Control":     "no-cache",
        "Connection":        "keep-alive",
    }
    return StreamingResponse(
        process_generator(),
        media_type="application/x-ndjson",
        headers=headers,
    )

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)