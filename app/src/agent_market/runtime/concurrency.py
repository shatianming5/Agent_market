from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple


class TaskCancelledError(RuntimeError):
    """Raised when execution is cancelled via the cancellation token."""


class CancellationToken:
    """Cooperative cancellation token built on top of threading.Event."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        """Wait for cancellation up to timeout seconds. Returns True if cancelled."""
        return self._event.wait(timeout)

    def throw_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise TaskCancelledError("operation cancelled")


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 1
    delay: float = 1.0
    max_delay: float = 60.0
    backoff: float = 2.0
    jitter: float = 0.25

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]], *, default_attempts: int = 1) -> "RetryPolicy":
        if not cfg:
            return cls(attempts=max(1, default_attempts))
        attempts = int(cfg.get("attempts") or cfg.get("max_attempts") or default_attempts or 1)
        delay = float(cfg.get("delay") or cfg.get("initial_delay") or 1.0)
        max_delay = float(cfg.get("max_delay") or max(delay, 1.0))
        backoff = float(cfg.get("backoff") or cfg.get("multiplier") or 2.0)
        jitter = float(cfg.get("jitter") or cfg.get("max_jitter") or 0.25)
        return cls(attempts=max(1, attempts), delay=max(0.0, delay), max_delay=max(delay, max_delay), backoff=max(1.0, backoff), jitter=max(0.0, jitter))


class RateLimiter:
    """Minimal token-based limiter ensuring a minimum interval between acquires."""

    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        self._interval = interval_seconds
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self, token: Optional[CancellationToken] = None) -> None:
        while True:
            if token:
                token.throw_if_cancelled()
            with self._lock:
                now = time.monotonic()
                wait = self._next_time - now
                if wait <= 0:
                    self._next_time = now + self._interval
                    return
            sleep_for = min(wait, 0.25) if wait > 0 else self._interval
            if token and token.wait(sleep_for):
                raise TaskCancelledError("cancelled while waiting for rate limiter slot")
            else:
                time.sleep(sleep_for)


@dataclass
class TaskSpec:
    name: str
    func: Callable[..., Any]
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    retry: Optional[RetryPolicy] = None
    rate_limit: Optional[str] = None


@dataclass
class TaskResult:
    name: str
    success: bool
    result: Any = None
    error: Optional[BaseException] = None
    attempts: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


class TaskScheduler:
    """Threaded task scheduler offering concurrency, rate limiting, and retries."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        rate_limiters: Optional[Dict[str, RateLimiter]] = None,
        default_retry: Optional[RetryPolicy] = None,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be >= 1")
        self._max_workers = max_workers
        self._rate_limiters = rate_limiters if rate_limiters is not None else {}
        self._default_retry = default_retry or RetryPolicy(attempts=1)

    def run(
        self,
        tasks: Iterable[TaskSpec],
        *,
        cancellation_token: Optional[CancellationToken] = None,
        fail_fast: bool = True,
    ) -> List[TaskResult]:
        tasks = list(tasks)
        if not tasks:
            return []
        token = cancellation_token or CancellationToken()
        results: List[Optional[TaskResult]] = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_map = {
                pool.submit(self._execute_task, idx, task, token): idx for idx, task in enumerate(tasks)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    result = future.result()
                except TaskCancelledError as exc:
                    result = TaskResult(
                        name=tasks[idx].name,
                        success=False,
                        error=exc,
                        attempts=0,
                        started_at=time.monotonic(),
                        finished_at=time.monotonic(),
                    )
                except BaseException as exc:  # pragma: no cover - unlikely, defensive
                    result = TaskResult(
                        name=tasks[idx].name,
                        success=False,
                        error=exc,
                        attempts=0,
                        started_at=time.monotonic(),
                        finished_at=time.monotonic(),
                    )
                results[idx] = result
                if fail_fast and not result.success:
                    token.cancel()

        return [r for r in results if r is not None]

    def _execute_task(
        self,
        index: int,
        task: TaskSpec,
        token: CancellationToken,
    ) -> TaskResult:
        token.throw_if_cancelled()
        retry_policy = task.retry or self._default_retry
        attempts = 0
        delay = max(0.0, retry_policy.delay)
        start_time = time.monotonic()
        last_error: Optional[BaseException] = None
        limiter = self._rate_limiters.get(task.rate_limit) if task.rate_limit else None

        while attempts < retry_policy.attempts:
            token.throw_if_cancelled()
            attempts += 1
            if limiter:
                limiter.acquire(token)
            call_started = time.monotonic()
            try:
                result = task.func(*task.args, **task.kwargs)
                return TaskResult(
                    name=task.name,
                    success=True,
                    result=result,
                    attempts=attempts,
                    started_at=start_time,
                    finished_at=time.monotonic(),
                )
            except TaskCancelledError as exc:
                token.cancel()
                return TaskResult(
                    name=task.name,
                    success=False,
                    error=exc,
                    attempts=attempts,
                    started_at=start_time,
                    finished_at=time.monotonic(),
                )
            except BaseException as exc:  # noqa: broad-except
                last_error = exc
                if attempts >= retry_policy.attempts:
                    break
                self._sleep_with_cancel(token, min(retry_policy.max_delay, delay + random.uniform(0, retry_policy.jitter)))
                delay = min(retry_policy.max_delay, delay * retry_policy.backoff if retry_policy.backoff else delay or retry_policy.delay)

        return TaskResult(
            name=task.name,
            success=False,
            error=last_error,
            attempts=attempts,
            started_at=start_time,
            finished_at=time.monotonic(),
        )

    @staticmethod
    def _sleep_with_cancel(token: CancellationToken, seconds: float) -> None:
        end_time = time.monotonic() + max(0.0, seconds)
        while True:
            token.throw_if_cancelled()
            remaining = end_time - time.monotonic()
            if remaining <= 0:
                return
            step = min(0.2, remaining)
            if token.wait(step):
                raise TaskCancelledError("cancelled during backoff wait")


__all__ = [
    "CancellationToken",
    "RateLimiter",
    "RetryPolicy",
    "TaskCancelledError",
    "TaskResult",
    "TaskScheduler",
    "TaskSpec",
]
