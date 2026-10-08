"""lev loaded in-process with `lev.load()`. One engine per process, loaded once."""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from ..features import RequestFeatures
from .base import LEV_QUESTIONS, Signals, signals_from_answers

log = logging.getLogger("tokentriage")


class LevNotReady(RuntimeError):
    """lev is still loading; the router falls back for this call."""


class LevBusy(RuntimeError):
    """An earlier decision timed out and is still running; the router falls back instead of queueing behind it."""


class LevLocalClassifier:
    name = "lev-local"

    _engine = None
    _load_error: BaseException | None = None
    _lock = threading.Lock()
    _loader: threading.Thread | None = None

    def __init__(self, checkpoint: str = "interfaze-ai/lev", timeout_s: float = 1.5, block_on_load: bool = False):
        self.checkpoint = checkpoint
        self.timeout_s = timeout_s
        self.block_on_load = block_on_load
        # One worker: torch forwards are serialized, and a timed-out call does not
        # pile up more work behind it.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokentriage")
        self._abandoned: Future | None = None  # the timed-out decision the worker is still running
        self._abandon_lock = threading.Lock()  # not the class-level load lock, which is held while lev loads

    # -- loading -------------------------------------------------------------

    def _load(self) -> None:
        cls = type(self)
        try:
            import lev
        except ImportError as exc:
            cls._load_error = ImportError(
                "lev is not installed. It is a separate prerequisite (not on PyPI): "
                "pip install \"lev[serve] @ git+https://github.com/InterfazeAI/lev#subdirectory=packages/lev\" "
                "(Python 3.12+). Until then tokentriage routes with its heuristic."
            )
            cls._load_error.__cause__ = exc
            return
        try:
            _warn_device()
            start = time.perf_counter()
            engine = lev.load(self.checkpoint)
            cls._engine = engine
            log.info("tokentriage: lev loaded from %s in %.1fs", self.checkpoint, time.perf_counter() - start)
        except BaseException as exc:  # noqa: BLE001 - surfaced through _load_error
            cls._load_error = exc
            log.warning("tokentriage: lev failed to load (%s); routing with the heuristic", exc)

    def warmup(self) -> None:
        """Load lev now, blocking. Call at app startup to avoid first-request latency."""
        with self._lock:
            if type(self)._engine is None and type(self)._load_error is None:
                self._load()
        if type(self)._load_error:
            raise type(self)._load_error

    def _ensure_loading(self) -> None:
        cls = type(self)
        if cls._engine is not None or cls._load_error is not None:
            return
        # A missing package is known instantly; report it as such rather than "still loading".
        if importlib.util.find_spec("lev") is None:
            self._load()
            return
        if self.block_on_load:
            # Block and wait for LEV to fully load
            with self._lock:
                if cls._engine is None and cls._load_error is None:
                    log.info("tokentriage: Loading LEV (blocking until ready)...")
                    self._load()
            return
        with self._lock:
            if cls._loader is None:
                cls._loader = threading.Thread(target=self._load, name="tokentriage-load", daemon=True)
                cls._loader.start()

    @property
    def ready(self) -> bool:
        return type(self)._engine is not None

    # -- classify ------------------------------------------------------------

    def classify(self, features: RequestFeatures, state: str) -> Signals:
        self._ensure_loading()
        log.debug("tokentriage: After _ensure_loading(), engine=%s, load_error=%s", type(self)._engine is not None, type(self)._load_error)
        cls = type(self)
        if cls._load_error is not None:
            raise cls._load_error
        if cls._engine is None:
            raise LevNotReady("lev is still loading")
        with self._abandon_lock:
            stuck = self._abandoned
            if stuck is not None and not stuck.done():
                raise LevBusy("an earlier lev decision timed out and is still running")
            self._abandoned = None
        start = time.perf_counter()
        future = self._pool.submit(cls._engine.system_one, state, LEV_QUESTIONS)
        try:
            response = future.result(timeout=self.timeout_s)
        except FutureTimeout:
            with self._abandon_lock:
                self._abandoned = future
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.debug("tokentriage: LEV classify - state_len=%d, elapsed_ms=%.0f, timeout_s=%.1f", len(state), elapsed_ms, self.timeout_s)
        return signals_from_answers(response.answers, self.name, elapsed_ms)


def _warn_device() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        log.warning(
            "tokentriage: no CUDA GPU found; lev runs on CPU and each decision may take seconds. "
            "Raise timeout_s, or run `lev serve` on a GPU and use backend='lev-http'."
        )
