
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent_market.connectors import REGISTRY, ConnectorError

from .db import ConnectorRecord, ConnectorTestLog, DB
from .secrets import SecretsManager

if TYPE_CHECKING:  # pragma: no cover
    from .job_manager import JobManager
    from .triggers import TriggerManager


class ConnectorService:
    def __init__(
        self,
        db: DB,
        secrets: SecretsManager,
        jobs: Optional['JobManager'] = None,
        triggers: Optional['TriggerManager'] = None,
    ) -> None:
        self._db = db
        self._secrets = secrets
        self._jobs = jobs
        self._triggers = triggers

    def list_types(self) -> List[str]:
        return REGISTRY.list_types()

    def list_connectors(self) -> List[ConnectorRecord]:
        return self._db.list_connectors()

    def get_connector(self, connector_id: str) -> Optional[ConnectorRecord]:
        return self._db.get_connector(connector_id)

    def list_test_logs(self, connector_id: str, limit: int = 20) -> List[ConnectorTestLog]:
        return self._db.list_connector_tests(connector_id, limit=limit)

    def create_connector(
        self,
        *,
        name: str,
        connector_type: str,
        config: Dict[str, Any],
        credentials: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        actor: Optional[str] = None,
    ) -> ConnectorRecord:
        if connector_type not in REGISTRY.list_types():
            raise ConnectorError(f"unsupported connector type '{connector_type}'")
        secret_scope = f"connector:{connector_type}"
        secret_name = uuid.uuid4().hex[:12]
        self._secrets.set_secret(secret_scope, secret_name, json.dumps(credentials), actor=actor)
        record = self._db.create_connector(
            name,
            connector_type,
            config,
            secret_scope=secret_scope,
            secret_name=secret_name,
            status="created",
            metadata=metadata,
        )
        return record

    def update_connector(
        self,
        connector_id: str,
        *,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        actor: Optional[str] = None,
    ) -> Optional[ConnectorRecord]:
        record = self._db.get_connector(connector_id)
        if not record:
            return None
        updates: Dict[str, Any] = {}
        if name is not None:
            updates['name'] = name
        if config is not None:
            updates['config'] = config
        if metadata is not None:
            updates['metadata'] = metadata
        if credentials is not None:
            self._secrets.set_secret(record.secret_scope, record.secret_name, json.dumps(credentials), actor=actor)
        if updates:
            record = self._db.update_connector(connector_id, **updates)
        return record

    def delete_connector(self, connector_id: str, actor: Optional[str] = None) -> None:
        record = self._db.get_connector(connector_id)
        if not record:
            return
        self._db.delete_connector(connector_id)
        self._secrets.delete_secret(record.secret_scope, record.secret_name, actor=actor)

    def test_connector(self, connector_id: str) -> Dict[str, Any]:
        record = self._db.get_connector(connector_id)
        if not record:
            raise ConnectorError("connector not found")
        secret = self._secrets.get_secret(record.secret_scope, record.secret_name, reveal=True)
        if not secret or secret.value in (None, "***"):
            raise ConnectorError("connector credentials missing")
        credentials = json.loads(secret.value)
        connector = REGISTRY.create(record.type, record.config, credentials)
        try:
            result = connector.test_connection()
            self._db.log_connector_test(record.id, "success", result)
            status = "healthy" if result.get("success") else "unknown"
            self._db.update_connector_status(connector_id, status=status, tested_at=datetime.now(timezone.utc).isoformat())
            return result
        except ConnectorError as exc:
            payload = {"error": str(exc)}
            self._db.log_connector_test(record.id, "error", payload)
            self._db.update_connector_status(connector_id, status="error", tested_at=datetime.now(timezone.utc).isoformat())
            raise

    def run_action(self, connector_id: str) -> Dict[str, Any]:
        record = self._db.get_connector(connector_id)
        if not record:
            raise ConnectorError("connector not found")
        metadata = record.metadata or {}
        if "trigger_id" in metadata:
            if not self._triggers:
                raise ConnectorError("trigger manager unavailable")
            payload = metadata.get("trigger_payload")
            payload_dict = payload if isinstance(payload, dict) else {}
            job_id = self._triggers.trigger_now(metadata["trigger_id"], payload=payload_dict)
            return {"type": "trigger", "job_id": job_id}
        if "job_command" in metadata:
            if not self._jobs:
                raise ConnectorError("job manager unavailable")
            command = metadata["job_command"]
            if not isinstance(command, list) or not command:
                raise ConnectorError("job_command must be a non-empty list")
            cmd = [str(part) for part in command]
            cwd = metadata.get("job_cwd")
            env = metadata.get("job_env")
            env_dict = {str(k): str(v) for k, v in (env or {}).items()} if env else None
            job_id = self._jobs.start(cmd, cwd=Path(cwd) if cwd else None, env=env_dict)
            return {"type": "job", "job_id": job_id}
        raise ConnectorError("no action configured for connector")


__all__ = ["ConnectorService"]
