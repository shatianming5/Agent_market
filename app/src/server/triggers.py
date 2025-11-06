from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from croniter import croniter

try:
    from web3 import Web3  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Web3 = None  # type: ignore

from .db import DB, TriggerRecord
from .job_manager import JobManager

logger = logging.getLogger(__name__)


def _ensure_list(command: Any) -> list[str]:
    if isinstance(command, (list, tuple)):
        return [str(part) for part in command]
    if isinstance(command, str):
        return [command]
    raise ValueError("trigger config requires 'command' as list or string")


def _normalize_env(values: Dict[str, Any]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in values.items()}


class TriggerBase:
    def __init__(self, record: TriggerRecord, manager: "TriggerManager") -> None:
        self.record = record
        self.manager = manager

    @property
    def enabled(self) -> bool:
        return self.record.enabled

    def tick(self, now: datetime) -> None:
        return

    def refresh(self, record: TriggerRecord) -> None:
        self.record = record

    def stop(self) -> None:  # pragma: no cover - hooks for subclasses needing cleanup
        return


class WebhookTrigger(TriggerBase):
    SECRET_HEADER = "x-trigger-secret"

    def handle(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        if not self.enabled:
            raise ValueError("trigger disabled")
        config = self.record.config
        secret = config.get("secret")
        if secret:
            provided = headers.get(self.SECRET_HEADER) or headers.get(self.SECRET_HEADER.title())
            if provided != secret:
                raise PermissionError("invalid webhook secret")
        return self.manager._start_job(self.record, payload=payload or {})


class CronTrigger(TriggerBase):
    def __init__(self, record: TriggerRecord, manager: "TriggerManager") -> None:
        super().__init__(record, manager)
        self._cron = None
        self._next_fire: Optional[datetime] = None
        self._prepare()

    def _prepare(self) -> None:
        expr = self.record.config.get("cron")
        if not expr:
            raise ValueError("cron trigger missing 'cron' expression")
        base = datetime.now(timezone.utc)
        self._cron = croniter(expr, base)
        self._next_fire = self._cron.get_next(datetime)

    def refresh(self, record: TriggerRecord) -> None:
        super().refresh(record)
        self._prepare()

    def tick(self, now: datetime) -> None:
        if not self.enabled or not self._cron or not self._next_fire:
            return
        # croniter returns naive datetime; align to aware
        next_fire = self._next_fire
        if next_fire.tzinfo is None:
            next_fire = next_fire.replace(tzinfo=timezone.utc)
        if now >= next_fire:
            self.manager._start_job(self.record, payload={"trigger_time": now.isoformat()})
            self._next_fire = self._cron.get_next(datetime)


class EvmTrigger(TriggerBase):
    def __init__(self, record: TriggerRecord, manager: "TriggerManager") -> None:
        super().__init__(record, manager)
        self._web3 = None
        self._next_poll: float = time.monotonic()
        self._last_block: Optional[int] = None
        self._poll_interval = float(record.config.get("poll_interval", 30))
        self._prepare()

    def _prepare(self) -> None:
        if Web3 is None:
            logger.warning("web3 library unavailable; EVM trigger disabled")
            return
        provider = self.record.config.get("provider") or self.record.config.get("provider_url")
        if not provider:
            raise ValueError("evm trigger missing 'provider_url'")
        self._web3 = Web3(Web3.HTTPProvider(provider, request_kwargs={"timeout": 10}))
        from_block = self.record.config.get("from_block")
        if isinstance(from_block, str) and from_block.lower() == "latest":
            try:
                self._last_block = int(self._web3.eth.block_number)  # type: ignore[union-attr]
            except Exception as exc:  # pragma: no cover - network errors
                logger.warning("failed to obtain current block: %s", exc)
                self._last_block = None
        elif from_block is not None:
            self._last_block = int(from_block)

    def refresh(self, record: TriggerRecord) -> None:
        super().refresh(record)
        self._poll_interval = float(record.config.get("poll_interval", 30))
        self._prepare()

    def tick(self, now: datetime) -> None:
        if not self.enabled or Web3 is None or self._web3 is None:
            return
        if time.monotonic() < self._next_poll:
            return
        self._next_poll = time.monotonic() + max(1.0, self._poll_interval)
        try:
            events = self._fetch_events()
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("EVM trigger %s failed to fetch logs: %s", self.record.id, exc)
            return
        if not events:
            return
        for event in events:
            payload = {
                "transactionHash": event.get("transactionHash"),
                "blockNumber": event.get("blockNumber"),
                "logIndex": event.get("logIndex"),
                "data": event.get("data"),
                "address": event.get("address"),
                "topics": event.get("topics"),
            }
            self.manager._start_job(self.record, payload=payload)
        last_block = events[-1].get("blockNumber")
        if last_block is not None:
            next_from = int(last_block) + 1
            cfg = dict(self.record.config)
            cfg["from_block"] = next_from
            self.manager._update_trigger_config(self.record.id, cfg)
            self._last_block = next_from

    def _fetch_events(self) -> list[Dict[str, Any]]:
        address = self.record.config.get("address")
        topics = self.record.config.get("topics")
        if address is None:
            raise ValueError("evm trigger missing 'address'")
        if topics is not None and not isinstance(topics, (list, tuple)):
            raise ValueError("evm trigger 'topics' must be list")
        params: Dict[str, Any] = {
            "address": address,
            "topics": topics,
        }
        from_block = self._last_block or self.record.config.get("from_block") or "latest"
        params["fromBlock"] = from_block
        params["toBlock"] = "latest"
        logs = self._web3.eth.get_logs(params)  # type: ignore[union-attr]
        return [dict(log) for log in logs]


TRIGGER_TYPE_MAP = {
    "webhook": WebhookTrigger,
    "cron": CronTrigger,
    "evm": EvmTrigger,
}


@dataclass
class ManagedTrigger:
    record: TriggerRecord
    handler: TriggerBase


class TriggerManager:
    def __init__(self, db: DB, jobs: JobManager, poll_interval: float = 5.0) -> None:
        self._db = db
        self._jobs = jobs
        self._poll_interval = poll_interval
        self._triggers: Dict[str, ManagedTrigger] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.reload()
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="trigger-manager", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            for managed in self._triggers.values():
                managed.handler.stop()

    def reload(self) -> None:
        records = self._db.list_triggers()
        with self._lock:
            # Stop old handlers not reused
            old_handlers = self._triggers
            self._triggers = {}
            for record in records:
                handler_class = TRIGGER_TYPE_MAP.get(record.type)
                if not handler_class:
                    logger.warning("unknown trigger type %s (id=%s)", record.type, record.id)
                    continue
                try:
                    if record.id in old_handlers and isinstance(old_handlers[record.id].handler, handler_class):
                        handler = old_handlers[record.id].handler
                        handler.refresh(record)
                    else:
                        handler = handler_class(record, self)
                except Exception as exc:
                    logger.warning("failed to initialise trigger %s: %s", record.id, exc)
                    continue
                self._triggers[record.id] = ManagedTrigger(record=record, handler=handler)

    def tick(self, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        with self._lock:
            triggers = list(self._triggers.values())
        for managed in triggers:
            try:
                managed.handler.tick(now)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("trigger %s tick failed: %s", managed.record.id, exc)

    def fire_webhook(self, trigger_id: str, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        trigger = self._get(trigger_id)
        if not isinstance(trigger.handler, WebhookTrigger):
            raise ValueError("trigger is not a webhook")
        return trigger.handler.handle(payload, headers)

    def trigger_now(self, trigger_id: str, payload: Optional[Dict[str, Any]] = None) -> str:
        trigger = self._get(trigger_id)
        if not trigger.record.enabled:
            raise ValueError("trigger disabled")
        return self._start_job(trigger.record, payload=payload or {})

    def refresh_single(self, trigger_id: str) -> None:
        record = self._db.get_trigger(trigger_id)
        if not record:
            with self._lock:
                existing = self._triggers.pop(trigger_id, None)
            if existing:
                existing.handler.stop()
            return
        with self._lock:
            handler_class = TRIGGER_TYPE_MAP.get(record.type)
            if not handler_class:
                logger.warning("unknown trigger type %s", record.type)
                return
            existing = self._triggers.get(record.id)
            if existing and isinstance(existing.handler, handler_class):
                existing.handler.refresh(record)
                existing.record = record
            else:
                if existing:
                    existing.handler.stop()
                try:
                    handler = handler_class(record, self)
                except Exception as exc:
                    logger.warning("failed to initialise trigger %s: %s", record.id, exc)
                    return
                self._triggers[record.id] = ManagedTrigger(record=record, handler=handler)

    def _loop(self) -> None:
        logger.info("Trigger manager loop started")
        while not self._stop_event.wait(self._poll_interval):
            self.tick()
        logger.info("Trigger manager loop stopped")

    def _start_job(self, record: TriggerRecord, payload: Optional[Dict[str, Any]] = None) -> str:
        config = record.config
        command = config.get("command")
        if not command:
            raise ValueError("trigger config missing 'command'")
        cmd = _ensure_list(command)
        cwd = config.get("cwd")
        cwd_path = Path(cwd) if cwd else None
        env = {
            "TRIGGER_ID": record.id,
            "TRIGGER_NAME": record.name,
            "TRIGGER_TYPE": record.type,
        }
        if payload is not None:
            env["TRIGGER_PAYLOAD"] = json.dumps(payload, ensure_ascii=False)
        env.update(_normalize_env(config.get("env", {})))
        try:
            job_id = self._jobs.start(cmd, cwd=cwd_path, env=env)
        except Exception as exc:
            logger.warning("failed to start job for trigger %s: %s", record.id, exc)
            raise
        self._db.update_trigger_last_fired(record.id)
        return job_id

    def _update_trigger_config(self, trigger_id: str, config: Dict[str, Any]) -> None:
        updated = self._db.update_trigger(trigger_id, config=config)
        if updated:
            self.refresh_single(trigger_id)

    def _get(self, trigger_id: str) -> ManagedTrigger:
        with self._lock:
            trigger = self._triggers.get(trigger_id)
        if not trigger:
            record = self._db.get_trigger(trigger_id)
            if not record:
                raise KeyError(trigger_id)
            self.refresh_single(trigger_id)
            with self._lock:
                trigger = self._triggers.get(trigger_id)
                if not trigger:
                    raise KeyError(trigger_id)
        return trigger


__all__ = ["TriggerManager", "TriggerRecord", "CronTrigger", "WebhookTrigger", "EvmTrigger"]
