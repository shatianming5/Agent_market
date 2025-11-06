import time

import pytest

from agent_market.runtime import (
    CancellationToken,
    RateLimiter,
    RetryPolicy,
    TaskCancelledError,
    TaskScheduler,
    TaskSpec,
)


def test_scheduler_runs_tasks_in_parallel():
    tracker = []

    def make_task(name: str, delay: float) -> TaskSpec:
        def runner() -> str:
            time.sleep(delay)
            tracker.append((name, time.monotonic()))
            return name

        return TaskSpec(name=name, func=runner)

    scheduler = TaskScheduler(max_workers=2)
    start = time.monotonic()
    scheduler.run([make_task("a", 0.2), make_task("b", 0.2), make_task("c", 0.2)])
    duration = time.monotonic() - start

    assert len(tracker) == 3
    # With concurrency=2 the total runtime should be noticeably less than sequential (0.6s)
    assert duration < 0.5


def test_scheduler_retries_until_success():
    counter = {"value": 0}

    def flaky() -> str:
        counter["value"] += 1
        if counter["value"] < 3:
            raise RuntimeError("boom")
        return "ok"

    policy = RetryPolicy(attempts=3, delay=0.01, max_delay=0.02, jitter=0.0)
    scheduler = TaskScheduler()
    result = scheduler.run([TaskSpec(name="flaky", func=flaky, retry=policy)])[0]

    assert result.success is True
    assert result.result == "ok"
    assert result.attempts == 3


def test_scheduler_rate_limiter_spacing():
    limiter = RateLimiter(interval_seconds=0.05)
    timestamps = []

    def job() -> None:
        timestamps.append(time.monotonic())

    scheduler = TaskScheduler(max_workers=3, rate_limiters={"api": limiter})
    scheduler.run(
        [TaskSpec(name=f"rate-{idx}", func=job, rate_limit="api") for idx in range(3)],
        fail_fast=False,
    )

    assert len(timestamps) == 3
    spacing = [s - timestamps[0] for s in timestamps[1:]]
    assert all(delta >= 0.05 - 0.01 for delta in spacing)  # allow small scheduling jitter


def test_scheduler_fail_fast_triggers_cancellation():
    scheduler = TaskScheduler(max_workers=1)

    def failing() -> None:
        raise RuntimeError("boom")

    def should_cancel() -> None:
        time.sleep(0.1)

    results = scheduler.run(
        [
            TaskSpec(name="fail", func=failing),
            TaskSpec(name="cancelled", func=should_cancel),
        ],
        fail_fast=True,
    )

    assert len(results) == 2
    assert results[0].success is False
    assert isinstance(results[0].error, RuntimeError)
    assert results[1].success is False
    assert isinstance(results[1].error, TaskCancelledError)


def test_scheduler_respects_external_cancellation():
    scheduler = TaskScheduler(max_workers=1)
    token = CancellationToken()

    def long_task() -> None:
        time.sleep(0.2)

    future = scheduler.run([TaskSpec(name="long", func=long_task)], cancellation_token=token, fail_fast=False)
    assert future[0].success is True

    # Cancellation before scheduling should short-circuit immediately.
    token.cancel()
    results = scheduler.run([TaskSpec(name="cancelled", func=long_task)], cancellation_token=token)
    assert len(results) == 1
    assert results[0].success is False
    assert isinstance(results[0].error, TaskCancelledError)
