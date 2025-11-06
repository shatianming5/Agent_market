from __future__ import annotations

import base64
import logging
from typing import Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken  # type: ignore

from .config import Settings
from .db import AuditRecord, DB, SecretRecord

logger = logging.getLogger(__name__)


class _IdentityCipher:
    """Fallback cipher that leaves values as-is (used when no master key provided)."""

    def encrypt(self, value: bytes) -> bytes:
        return value

    def decrypt(self, value: bytes) -> bytes:
        return value


def _build_cipher(master_key: Optional[str]) -> object:
    if not master_key:
        logger.warning("No SECRETS_MASTER_KEY configured; secrets will be stored in plaintext")
        return _IdentityCipher()
    key = master_key.strip()
    if len(key) == 32:
        key_bytes = base64.urlsafe_b64encode(key.encode("utf-8"))
    else:
        try:
            key_bytes = base64.urlsafe_b64decode(key)
        except Exception:
            logger.warning("SECRETS_MASTER_KEY is not valid base64; secrets will fallback to plaintext")
            return _IdentityCipher()
        if len(key_bytes) != 32:
            logger.warning("SECRETS_MASTER_KEY must decode to 32 bytes; using plaintext fallback")
            return _IdentityCipher()
        key_bytes = base64.urlsafe_b64encode(key_bytes)
    try:
        return Fernet(key_bytes)
    except Exception as exc:
        logger.warning("Failed to build Fernet cipher: %s", exc)
        return _IdentityCipher()


class SecretsManager:
    def __init__(self, db: DB, settings: Settings):
        self._db = db
        self._settings = settings
        self._cipher = _build_cipher(settings.secrets_master_key)
        self._allowed_scopes = set(settings.secret_scopes or [])
        self._token = settings.secret_admin_token
        self._read_token = settings.secret_read_token

    def _ensure_scope(self, scope: str) -> None:
        if self._allowed_scopes and scope not in self._allowed_scopes:
            raise PermissionError(f"scope '{scope}' is not permitted")

    def _encrypt(self, value: str) -> str:
        data = value.encode("utf-8")
        encrypted = self._cipher.encrypt(data)
        return encrypted.decode("utf-8")

    def _decrypt(self, value: str) -> str:
        try:
            data = value.encode("utf-8")
            decrypted = self._cipher.decrypt(data)
            return decrypted.decode("utf-8")
        except InvalidToken:
            logger.warning("Failed to decrypt secret; returning masked value")
            return "***"
        except Exception:
            logger.warning("Unexpected error decrypting secret")
            return "***"

    def validate_token(self, token: Optional[str]) -> None:
        if not self._token:
            return
        if token != self._token:
            raise PermissionError("invalid secret token")

    def validate_read_token(self, token: Optional[str]) -> None:
        tokens = [t for t in (self._token, self._read_token) if t]
        if not tokens:
            return
        if token in tokens:
            return
        raise PermissionError("invalid secret token")

    def set_secret(self, scope: str, name: str, value: str, actor: Optional[str] = None) -> SecretRecord:
        self._ensure_scope(scope)
        encrypted = self._encrypt(value)
        record = self._db.set_secret(scope, name, encrypted)
        self._db.log_audit(actor=actor, action="secret.set", resource_type="secret", resource_id=f"{scope}:{name}", payload={"masked": True})
        record.value = "***"
        return record

    def get_secret(self, scope: str, name: str, *, reveal: bool = False) -> Optional[SecretRecord]:
        self._ensure_scope(scope)
        include_value = reveal
        record = self._db.get_secret(scope, name, include_value=include_value)
        if not record:
            return None
        if reveal and record.value not in ("***", None):
            record.value = self._decrypt(record.value)
        else:
            record.value = "***"
        return record

    def list_secrets(self, scope: Optional[str] = None) -> List[SecretRecord]:
        if scope:
            self._ensure_scope(scope)
        secrets = self._db.list_secrets(scope, include_values=False)
        for secret in secrets:
            secret.value = "***"
        return secrets

    def delete_secret(self, scope: str, name: str, actor: Optional[str] = None) -> None:
        self._ensure_scope(scope)
        self._db.delete_secret(scope, name)
        self._db.log_audit(actor=actor, action="secret.delete", resource_type="secret", resource_id=f"{scope}:{name}", payload=None)


__all__ = ["SecretsManager"]
