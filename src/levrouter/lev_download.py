"""Fetch lev and its base model into the Hugging Face cache, with visible progress.

lev is a LoRA adapter (~0.2 GB) on top of a base model (~9.3 GB), so the first run has to
download both. This blocks until every file is on disk and prints a live percentage, so the
caller never starts loading or routing on a half-downloaded model.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from tqdm import tqdm

# lev loads the base config and weights only; its tokenizer comes from the adapter repo.
BASE_PATTERNS = ["*.safetensors", "model.safetensors.index.json", "config.json"]


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


class _Silent(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


class _Progress:
    """Bytes downloaded across all files of one repo, rendered as a single updating line."""

    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(total, 1)
        self.done = 0
        self._start = time.monotonic()
        self._last_print = 0.0
        self._full = False
        self._lock = threading.Lock()

    def add(self, n: int) -> None:
        with self._lock:
            self.done = min(self.done + n, self.total)
            self._render()

    def finish(self) -> None:
        with self._lock:
            self.done = self.total
            self._render()

    def _render(self) -> None:
        now = time.monotonic()
        if self._full or not self.done:
            return
        if now - self._last_print < 0.5 and self.done < self.total:
            return
        self._last_print = now
        self._full = self.done >= self.total
        rate = self.done / max(now - self._start, 1e-6)
        eta = f"{(self.total - self.done) / rate:.0f}s" if rate > 1e3 and not self._full else "--"
        print(
            f"\r   {self.label}: {100 * self.done / self.total:5.1f}%  "
            f"{_gb(self.done)} / {_gb(self.total)}  {rate / 1e6:.0f} MB/s  ETA {eta}   ",
            end="",
            flush=True,
        )

    def bar_class(self) -> type:
        progress = self

        class Bar(tqdm):
            # Only the per-file byte bars count; the outer "Fetching N files" bar is ignored.
            def __init__(self, *args, **kwargs):
                self._is_bytes = kwargs.get("unit") == "B"
                initial = kwargs.get("initial", 0) or 0
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)
                if self._is_bytes and initial:
                    progress.add(initial)

            def update(self, n=1):
                if self._is_bytes:
                    progress.add(n)
                return super().update(n)

        return Bar


def _fetch(label: str, repo: str, patterns: list[str] | None) -> str:
    from huggingface_hub import snapshot_download

    try:
        plan = snapshot_download(repo, allow_patterns=patterns, dry_run=True, tqdm_class=_Silent)
    except Exception:  # noqa: BLE001 - offline or hub unreachable: trust the cache if it has the repo
        return snapshot_download(repo, allow_patterns=patterns, local_files_only=True)

    todo = [f for f in plan if f.will_download]
    if not todo:
        print(f"   ✅ {label} ({repo}): already in the local cache")
        return snapshot_download(repo, allow_patterns=patterns, local_files_only=True)

    total = sum(f.file_size for f in todo)
    print(f"📥 {label} ({repo}) is not cached: downloading {_gb(total)} in {len(todo)} file(s).")
    print("   Nothing else runs until this finishes.")
    progress = _Progress(label, total)
    path = snapshot_download(repo, allow_patterns=patterns, tqdm_class=progress.bar_class())
    progress.finish()
    print(f"\n   ✅ {label} downloaded")
    return path


def ensure_lev_cached(checkpoint: str = "interfaze-ai/lev") -> bool:
    """Block until the lev adapter and its base model are in the local cache.

    Returns False (after printing why) if either cannot be obtained, e.g. no network on first run.
    """
    print(f"🔎 levrouter: checking the local Hugging Face cache for {checkpoint} and its base model...")
    try:
        adapter_dir = _fetch("lev adapter", checkpoint, None)
        manifest = json.loads((Path(adapter_dir) / "lev_release.json").read_text())
        _fetch("base model", manifest["base_model"], BASE_PATTERNS)
    except Exception as exc:  # noqa: BLE001 - reported to the user, who falls back to the heuristic
        print(f"⚠️  levrouter: could not get the lev model files: {exc}")
        return False
    return True
