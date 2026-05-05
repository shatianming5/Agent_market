"""WorldQuant BRAIN HTTP client.

Login → JWT session cookie. All endpoints behind a session-level Semaphore
so multiple callers (agent + scan) sharing the same WQSession never exceed
the global concurrency cap.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests

from .dtypes import AlphaCandidate, AlphaSettings, SimulationResult

logger = logging.getLogger(__name__)

WQ_API_BASE = os.environ.get("WQ_API_BASE", "https://api.worldquantbrain.com")

_POLL_INTERVAL = 10.0
_RETRY_DELAYS = (5, 15, 30, 60)


class WQSession(requests.Session):
    """requests.Session with WQ auth, exponential retry, and concurrency cap."""

    def __init__(
        self,
        email: str,
        password: str,
        *,
        api_base: str = WQ_API_BASE,
        max_concurrent: int = 3,
    ) -> None:
        super().__init__()
        self._email = email
        self._password = password
        self._api_base = api_base.rstrip("/")
        self._logged_in = False
        self._max_concurrent = max_concurrent
        self._login_lock = threading.Lock()
        self._global_sem = threading.Semaphore(max_concurrent)

    def login(self) -> None:
        url = f"{self._api_base}/authentication"
        resp = super().post(url, auth=(self._email, self._password), timeout=30)
        if resp.status_code == 401:
            raise PermissionError("WQ authentication failed: invalid credentials")
        resp.raise_for_status()
        self._logged_in = True
        logger.info("WQ login successful")

    def _ensure_login(self) -> None:
        if not self._logged_in:
            with self._login_lock:
                if not self._logged_in:
                    self.login()

    def _request_with_retry(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self._ensure_login()
        last_exc: Optional[Exception] = None
        for attempt, delay in enumerate([0, *_RETRY_DELAYS]):
            if delay:
                time.sleep(delay)
            resp = super().request(method, url, **kwargs)
            if resp.status_code == 401:
                logger.warning("Session expired on %s, re-logging in", url)
                self._logged_in = False
                with self._login_lock:
                    if not self._logged_in:
                        try:
                            self.login()
                        except Exception as exc:
                            raise PermissionError(f"Re-login failed for {url}: {exc}") from exc
                resp = super().request(method, url, **kwargs)
                if resp.status_code == 401:
                    raise PermissionError(f"Still 401 after re-login: {url}")
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "60"))
                logger.warning("Rate limited; sleeping %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            if resp.status_code >= 500 and attempt < len(_RETRY_DELAYS):
                continue
            try:
                resp.raise_for_status()
                return resp
            except Exception as exc:
                last_exc = exc
                if attempt < len(_RETRY_DELAYS):
                    continue
                raise
        raise RuntimeError(f"All retries exhausted for {method} {url}: {last_exc}")

    def submit_simulation(self, expr: str, settings: AlphaSettings) -> str:
        payload = {"type": "REGULAR", "settings": settings.to_api_dict(), "regular": expr}
        url = f"{self._api_base}/simulations"
        with self._global_sem:
            resp = self._request_with_retry("POST", url, json=payload, timeout=60)
        sim_url = resp.headers.get("Location") or resp.json().get("url", "")
        if not sim_url:
            raise RuntimeError(f"No simulation URL returned; body={resp.text[:200]}")
        if not sim_url.startswith("http"):
            sim_url = f"{self._api_base}{sim_url}"
        return sim_url

    def poll_simulation(self, sim_url: str, *, timeout: float = 600.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        while True:
            resp = self._request_with_retry("GET", sim_url, timeout=30)
            data = resp.json()
            status = str(data.get("status", "")).upper()
            if status in ("COMPLETE", "ERROR", "FAILED", "UNKNOWN"):
                return data
            if time.time() > deadline:
                raise TimeoutError(f"Simulation timed out after {timeout}s: {sim_url}")
            time.sleep(_POLL_INTERVAL)

    def parse_sim_result(self, raw: dict[str, Any]) -> SimulationResult:
        status = str(raw.get("status", "UNKNOWN")).upper()
        alpha_field = raw.get("alpha")
        if isinstance(alpha_field, str):
            alpha_id = alpha_field
            is_data: dict[str, Any] = {}
        elif isinstance(alpha_field, dict):
            alpha_id = alpha_field.get("id") or raw.get("id")
            is_data = alpha_field.get("is") or {}
        else:
            alpha_id = raw.get("id") or raw.get("alphaId")
            is_data = raw.get("is") or {}

        if status in ("ERROR", "FAILED"):
            err = raw.get("error") or raw.get("message") or "unknown error"
            return SimulationResult(status=status, error=str(err), alpha_id=alpha_id)

        def _f(key: str) -> Optional[float]:
            v = is_data.get(key)
            return float(v) if v is not None else None

        return SimulationResult(
            sharpe=_f("sharpe"),
            fitness=_f("fitness"),
            returns=_f("returns"),
            turnover=_f("turnover"),
            drawdown=_f("drawdown"),
            long_count=is_data.get("longCount"),
            short_count=is_data.get("shortCount"),
            alpha_id=str(alpha_id) if alpha_id else None,
            status=status,
        )

    def fetch_alpha_metrics(self, alpha_id: str) -> SimulationResult:
        url = f"{self._api_base}/alphas/{alpha_id}"
        with self._global_sem:
            data = self._request_with_retry("GET", url, timeout=30).json()
        is_data = data.get("is") or {}
        status = "COMPLETE" if is_data else "UNKNOWN"

        def _f(key: str) -> Optional[float]:
            v = is_data.get(key)
            return float(v) if v is not None else None

        return SimulationResult(
            sharpe=_f("sharpe"),
            fitness=_f("fitness"),
            returns=_f("returns"),
            turnover=_f("turnover"),
            drawdown=_f("drawdown"),
            long_count=is_data.get("longCount"),
            short_count=is_data.get("shortCount"),
            alpha_id=alpha_id,
            status=status,
            checks=is_data.get("checks") or [],
        )

    def simulate_and_parse(
        self, expr: str, settings: AlphaSettings, *, timeout: float = 600.0
    ) -> SimulationResult:
        sim_url = self.submit_simulation(expr, settings)
        raw = self.poll_simulation(sim_url, timeout=timeout)
        result = self.parse_sim_result(raw)
        if result.alpha_id and result.sharpe is None:
            try:
                result = self.fetch_alpha_metrics(result.alpha_id)
            except Exception as exc:
                logger.warning("Failed to fetch alpha metrics for %s: %s", result.alpha_id, exc)
        return result

    def get_alpha(self, alpha_id: str) -> dict[str, Any]:
        url = f"{self._api_base}/alphas/{alpha_id}"
        with self._global_sem:
            return self._request_with_retry("GET", url, timeout=30).json()

    def list_my_alphas(self, limit: int = 100) -> list[dict[str, Any]]:
        url = f"{self._api_base}/alphas"
        with self._global_sem:
            try:
                resp = self._request_with_retry("GET", url, params={"limit": limit}, timeout=30)
                data = resp.json()
                if isinstance(data, list):
                    return data[:limit]
                return data.get("results", data.get("alphas", []))
            except Exception:
                return []

    def get_alpha_correlations(self, alpha_id: str) -> list[dict[str, Any]]:
        url = f"{self._api_base}/alphas/{alpha_id}/correlations"
        with self._global_sem:
            resp = self._request_with_retry("GET", url, timeout=30)
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("alphas", [])
        return []

    def submit_alpha(self, alpha_id: str, *, verify_after_sec: float = 30.0) -> dict[str, Any]:
        """Submit an alpha + verify final status.

        WQ's POST /alphas/{id}/submit returns 201 even when the alpha will
        ultimately be rejected by async server-side review (e.g. self-correlation
        > 0.7). To distinguish queued-but-rejected from queued-and-accepted, we
        sleep `verify_after_sec` seconds and re-fetch the alpha; a true success
        is `status: ACTIVE` after the wait.

        Returns dict with always-present fields:
            submitted: bool          — True only if verified ACTIVE
            verified_status: str     — ACTIVE / REJECTED / QUEUED / VERIFICATION_FAILED
            rejection_reasons: list  — failed `is.checks` entries (when REJECTED)
            initial_status_code: int — POST response status
            alpha_id: str
        """
        url = f"{self._api_base}/alphas/{alpha_id}/submit"
        with self._global_sem:
            resp = super().request("POST", url, timeout=30)

        if resp.status_code == 401:
            self._logged_in = False
            with self._login_lock:
                if not self._logged_in:
                    self.login()
            with self._global_sem:
                resp = super().request("POST", url, timeout=30)

        # Parse response body (best-effort)
        try:
            body = resp.json()
            if not isinstance(body, dict):
                body = {"raw": body}
        except (ValueError, AttributeError):
            body = {"text": (getattr(resp, "text", "") or "")[:500]}

        # Synchronous 4xx rejection — WQ has previous failure cached
        if resp.status_code >= 400 and resp.status_code != 429:
            checks = (body.get("is") or {}).get("checks") or []
            failed = [{"name": c.get("name"), "result": c.get("result"),
                       "limit": c.get("limit"), "value": c.get("value")}
                      for c in checks if c.get("result") == "FAIL"]
            return {
                "submitted": False,
                "alpha_id": alpha_id,
                "verified_status": "REJECTED",
                "rejection_reasons": failed,
                "initial_status_code": resp.status_code,
                "raw": body,
            }
        if resp.status_code == 429:
            raise RuntimeError(
                f"WQ submit throttled (HTTP 429) for alpha {alpha_id}: {body}"
            )
        # 2xx — submission queued; need to verify
        if verify_after_sec > 0:
            time.sleep(verify_after_sec)
            try:
                check_url = f"{self._api_base}/alphas/{alpha_id}"
                with self._global_sem:
                    check_resp = self._request_with_retry("GET", check_url, timeout=20)
                data = check_resp.json()
                actual_status = data.get("status") or "UNKNOWN"
                checks = (data.get("is") or {}).get("checks") or []
                failed = [{"name": c.get("name"), "result": c.get("result"),
                           "limit": c.get("limit"), "value": c.get("value")}
                          for c in checks if c.get("result") == "FAIL"]
                if actual_status == "ACTIVE":
                    return {
                        "submitted": True,
                        "alpha_id": alpha_id,
                        "verified_status": "ACTIVE",
                        "rejection_reasons": [],
                        "initial_status_code": resp.status_code,
                        "date_submitted": data.get("dateSubmitted"),
                    }
                # Not ACTIVE → rejection
                return {
                    "submitted": False,
                    "alpha_id": alpha_id,
                    "verified_status": actual_status,  # typically UNSUBMITTED
                    "rejection_reasons": failed,
                    "initial_status_code": resp.status_code,
                }
            except Exception as exc:
                return {
                    "submitted": None,
                    "alpha_id": alpha_id,
                    "verified_status": "VERIFICATION_FAILED",
                    "verification_error": str(exc)[:200],
                    "initial_status_code": resp.status_code,
                }
        # verify disabled
        return {
            "submitted": None,
            "alpha_id": alpha_id,
            "verified_status": "QUEUED",
            "initial_status_code": resp.status_code,
            "raw": body,
        }

    def get_alpha_status(self, alpha_id: str) -> dict[str, Any]:
        """Quick status check: is alpha submitted? Returns raw fields."""
        url = f"{self._api_base}/alphas/{alpha_id}"
        with self._global_sem:
            data = self._request_with_retry("GET", url, timeout=20).json()
        return {
            "alpha_id": alpha_id,
            "status": data.get("status"),
            "stage": data.get("stage"),
            "date_submitted": data.get("dateSubmitted"),
            "grade": data.get("grade"),
            "category": data.get("category"),
        }

    def batch_simulate(
        self,
        candidates: list[AlphaCandidate],
        *,
        timeout: float = 600.0,
    ) -> list[AlphaCandidate]:
        """Simulate concurrently. Concurrency capped by session-level Semaphore."""
        def _run(c: AlphaCandidate) -> AlphaCandidate:
            try:
                c.sim_result = self.simulate_and_parse(c.expr, c.settings, timeout=timeout)
            except Exception as exc:
                logger.warning("Simulation failed for %s: %s", c.candidate_id, exc)
                c.sim_result = SimulationResult(status="ERROR", error=str(exc))
            return c

        n_workers = min(len(candidates), self._max_concurrent) or 1
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run, c): c for c in candidates}
            results: list[AlphaCandidate] = []
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    logger.error("Unexpected error in batch_simulate: %s", exc)
        return results


def session_from_env() -> WQSession:
    email = os.environ.get("WQ_EMAIL", "")
    password = os.environ.get("WQ_PASSWORD", "")
    api_base = os.environ.get("WQ_API_BASE", WQ_API_BASE)
    max_concurrent = int(os.environ.get("WQ_MAX_CONCURRENT", "3"))
    if not email or not password:
        raise EnvironmentError(
            "WQ_EMAIL and WQ_PASSWORD must be set (in .env or environment)"
        )
    return WQSession(email, password, api_base=api_base, max_concurrent=max_concurrent)
