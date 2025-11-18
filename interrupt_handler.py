import os
import re
import asyncio
import logging
from typing import Iterable, Callable, Optional, List

# Configure basic logging for this module
logger = logging.getLogger("interrupt_handler")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s interrupt_handler: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def normalize_text(text: str) -> str:
    # Normalize unicode, lower, collapse whitespace, strip punctuation around words
    import unicodedata
    t = unicodedata.normalize("NFKC", text)
    t = t.lower()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(text: str) -> List[str]:
    t = normalize_text(text)
    return t.split() if t else []

class InterruptHandler:
    """
    Extension layer that filters filler-only ASR events when the agent is speaking.

    Usage:
      handler = InterruptHandler(agent, on_stop_callback=agent.stop, ...)
      agent.on("transcription", handler.on_transcript_event)
      agent.on("tts_start", handler.on_tts_start)
      agent.on("tts_end", handler.on_tts_end)
    """

    def __init__(
        self,
        agent,
        *,
        ignored_words: Optional[Iterable[str]] = None,
        stop_keywords: Optional[Iterable[str]] = None,
        confidence_threshold: float = 0.6,
        on_stop_callback: Optional[Callable[[], None]] = None,
        on_pause_callback: Optional[Callable[[], None]] = None,
    ):
        self.agent = agent
        # Keep sets normalized for fast comparison
        self.ignored_words = set(normalize_text(w) for w in (ignored_words or []))
        self.stop_keywords = set(normalize_text(w) for w in (stop_keywords or ["stop", "wait", "hold on", "no"]))
        self.confidence_threshold = float(os.getenv("INTERRUPT_CONFIDENCE_THRESHOLD", confidence_threshold))
        self.on_stop_callback = on_stop_callback or getattr(agent, "stop", None)
        self.on_pause_callback = on_pause_callback or getattr(agent, "pause_tts", None)
        self._speaking = False  # updated via tts start/end events
        self._lock = asyncio.Lock()
        logger.info("InterruptHandler initialized. ignored_words=%s stop_keywords=%s confidence_threshold=%.2f",
                    sorted(self.ignored_words), sorted(self.stop_keywords), self.confidence_threshold)

    # ---------- public API ----------
    async def update_ignored_words(self, new_list: Iterable[str]):
        """Dynamically update ignored words at runtime (async-safe)."""
        async with self._lock:
            self.ignored_words = set(normalize_text(w) for w in new_list)
            logger.info("Updated ignored_words -> %s", sorted(self.ignored_words))

    # These should be wired to tts start/stop events
    async def on_tts_start(self, *args, **kwargs):
        async with self._lock:
            self._speaking = True
            logger.debug("Agent speaking state -> True")

    async def on_tts_end(self, *args, **kwargs):
        async with self._lock:
            self._speaking = False
            logger.debug("Agent speaking state -> False")

    # Main handler for transcription events from ASR
    async def on_transcript_event(self, transcript: str, confidence: Optional[float] = None, raw_event: Optional[dict]=None):
        """
        transcript: the transcribed text segment (string)
        confidence: optional average confidence for the segment (0..1)
        raw_event: optional raw ASR event for logging/debugging
        """
        # quickly normalize and tokenize
        tokens = tokenize(transcript)
        # capture numeric/None safe value for confidence
        conf = float(confidence) if confidence is not None else None

        async with self._lock:
            speaking = self._speaking

        # Logging incoming event
        logger.debug("Transcript received: '%s' (tokens=%s) conf=%s speaking=%s", transcript, tokens, conf, speaking)

        # If mixed content contains any explicit stop keyword -> treat as valid interruption
        if self._contains_stop_keyword(tokens):
            logger.info("Valid interruption detected (stop keyword) in transcript: '%s'. Triggering stop.", transcript)
            await self._trigger_stop()
            self._log_valid_interruption(transcript, conf, raw_event)
            return {"action": "stop", "reason": "stop_keyword", "transcript": transcript}

        # If agent is not speaking -> always treat as valid user speech (don't ignore)
        if not speaking:
            logger.info("Agent was quiet. Treating transcript as user speech: '%s'", transcript)
            self._log_valid_interruption(transcript, conf, raw_event)
            return {"action": "pass_through", "reason": "agent_quiet", "transcript": transcript}

        # Agent is speaking -> decide whether to ignore
        # If confidence provided and is low, ignore (noise) unless stop keyword present
        if conf is not None and conf < self.confidence_threshold:
            logger.info("Ignoring low-confidence transcript while agent speaking (conf=%.2f < %.2f): '%s'",
                        conf, self.confidence_threshold, transcript)
            self._log_ignored_interruption(transcript, conf, raw_event)
            return {"action": "ignore", "reason": "low_confidence", "transcript": transcript}

        # Check if the transcript only contains filler words
        if self._is_filler_only(tokens):
            logger.info("Ignoring filler-only transcript while agent speaking: '%s'", transcript)
            self._log_ignored_interruption(transcript, conf, raw_event)
            return {"action": "ignore", "reason": "filler_only", "transcript": transcript}

        # Mixed content (filler + command) -> pass through and likely stop
        logger.info("Transcript appears to contain meaningful content while agent speaking: '%s'", transcript)
        self._log_valid_interruption(transcript, conf, raw_event)
        # Optionally, you can parse commands here. For now we trigger stop to be strict real-time.
        await self._trigger_stop()
        return {"action": "stop", "reason": "meaningful_content", "transcript": transcript}

    # ---------- helpers ----------
    def _contains_stop_keyword(self, tokens: List[str]) -> bool:
        # simple check: any contiguous tokens matching stop_keywords or exact token match
        if not tokens:
            return False
        joined = " ".join(tokens)
        # exact match or any of phrases
        for kw in self.stop_keywords:
            if kw in joined or kw == joined:
                return True
        # also single token matches
        return any(t in self.stop_keywords for t in tokens)

    def _is_filler_only(self, tokens: List[str]) -> bool:
        if not tokens:
            return True
        # if every token is in ignored_words set, it's filler-only
        return all(token in self.ignored_words for token in tokens)

    async def _trigger_stop(self):
        # real-time stop: call the provided callback (await if coroutine)
        if asyncio.iscoroutinefunction(self.on_stop_callback):
            try:
                await self.on_stop_callback()
            except Exception as e:
                logger.exception("Error calling async stop callback: %s", e)
        elif callable(self.on_stop_callback):
            try:
                self.on_stop_callback()
            except Exception as e:
                logger.exception("Error calling stop callback: %s", e)
        # ensure speaking flag turned off (best-effort)
        async with self._lock:
            self._speaking = False

    # Logging helpers for debugging
    def _log_ignored_interruption(self, transcript, conf, raw_event):
        logger.debug("Ignored interruption | transcript=%r | conf=%s | raw=%s", transcript, conf, raw_event)

    def _log_valid_interruption(self, transcript, conf, raw_event):
        logger.debug("Valid interruption | transcript=%r | conf=%s | raw=%s", transcript, conf, raw_event)

# Example: build default handler from env variables
def build_default_handler(agent, on_stop_callback=None):
    ignored = os.getenv("IGNORED_WORDS", "uh,umm,ummm,hmm,haan,huh").split(",")
    stop_keys = os.getenv("STOP_KEYWORDS", "stop,wait,hold on,no,wait one second").split(",")
    conf_thresh = float(os.getenv("INTERRUPT_CONFIDENCE_THRESHOLD", "0.6"))
    return InterruptHandler(agent,
                            ignored_words=ignored,
                            stop_keywords=stop_keys,
                            confidence_threshold=conf_thresh,
                            on_stop_callback=on_stop_callback)
