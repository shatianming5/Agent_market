"""WorldQuant BRAIN API client.

Authentication: HTTP Basic Auth → JWT session cookie.
All requests after login use the session cookie automatically.
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

_POLL_INTERVAL = 10.0    # seconds between status polls
_RETRY_DELAYS = (5, 15, 30, 60)  # exponential backoff for retries


class WQSession(requests.Session):
    """Thin wrapper around requests.Session that handles WQ auth and retries."""

    def __init__(
        self, email: str, password: str, *, api_base: str = WQ_API_BASE, max_concurrent: int = 3
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
        if not self._logged_in:  # fast-path (no lock)
            with self._login_lock:
                if not self._logged_in:  # double-checked under lock
                    self.login()

    def _request_with_retry(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self._ensure_login()
        for attempt, delay in enumerate([0, *_RETRY_DELAYS]):
            if delay:
                logger.debug("Retry %d/%d after %ds for %s", attempt, len(_RETRY_DELAYS), delay, url)
                time.sleep(delay)
            resp = super().request(method, url, **kwargs)
            # 401: re-login immediately under lock, doesn't consume a retry slot
            if resp.status_code == 401:
                logger.warning("Session expired, re-logging in")
                self._logged_in = False  # mark expired before lock so double-check works
                with self._login_lock:
                    if not self._logged_in:  # another thread may have already re-logged in
                        try:
                            self.login()
                        except Exception as exc:
                            raise PermissionError(f"Re-login failed for {url}: {exc}") from exc
                resp = super().request(method, url, **kwargs)
                if resp.status_code == 401:
                    raise PermissionError(f"Still 401 after re-login: {url}")
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "60"))
                logger.warning("Rate limited; sleeping %ds (attempt %d/%d)", wait, attempt + 1, len(_RETRY_DELAYS) + 1)
                time.sleep(wait)
                continue
            if resp.status_code >= 500 and attempt < len(_RETRY_DELAYS):
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"All {len(_RETRY_DELAYS) + 1} retries exhausted for {method} {url}")

    # ── Simulation ────────────────────────────────────────────────────────

    def submit_simulation(self, expr: str, settings: AlphaSettings) -> str:
        """Submit a simulation. Returns the polling URL (Location header)."""
        payload = {
            "type": "REGULAR",
            "settings": settings.to_api_dict(),
            "regular": expr,
        }
        url = f"{self._api_base}/simulations"
        resp = self._request_with_retry("POST", url, json=payload, timeout=60)
        sim_url = resp.headers.get("Location") or resp.json().get("url", "")
        if not sim_url:
            raise RuntimeError(f"No simulation URL returned; body={resp.text[:200]}")
        if not sim_url.startswith("http"):
            sim_url = f"{self._api_base}{sim_url}"
        logger.debug("Simulation submitted: %s", sim_url)
        return sim_url

    def poll_simulation(self, sim_url: str, *, timeout: float = 300.0) -> dict[str, Any]:
        """Poll until simulation completes. Returns raw result dict."""
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
        """Extract SimulationResult from raw poll response or alpha details response."""
        status = str(raw.get("status", "UNKNOWN")).upper()

        # After simulation completes, alpha may be a string ID or a nested dict
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
        """Fetch alpha details and parse metrics from /alphas/{id}."""
        url = f"{self._api_base}/alphas/{alpha_id}"
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
        self, expr: str, settings: AlphaSettings, *, timeout: float = 300.0
    ) -> SimulationResult:
        """Convenience: submit + poll + fetch alpha metrics."""
        sim_url = self.submit_simulation(expr, settings)
        raw = self.poll_simulation(sim_url, timeout=timeout)
        result = self.parse_sim_result(raw)
        # If alpha_id present but metrics missing, fetch them separately
        if result.alpha_id and result.sharpe is None:
            try:
                result = self.fetch_alpha_metrics(result.alpha_id)
            except Exception as exc:
                logger.warning("Failed to fetch alpha metrics for %s: %s", result.alpha_id, exc)
        return result

    # ── Alpha pool ────────────────────────────────────────────────────────

    def get_alpha(self, alpha_id: str) -> dict[str, Any]:
        url = f"{self._api_base}/alphas/{alpha_id}"
        return self._request_with_retry("GET", url, timeout=30).json()

    def list_my_alphas(self, limit: int = 100) -> list[dict[str, Any]]:
        url = f"{self._api_base}/alphas"
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
        resp = self._request_with_retry("GET", url, timeout=30)
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("alphas", [])
        return []

    def submit_alpha(self, alpha_id: str) -> dict[str, Any]:
        """Submit a simulated alpha to the permanent pool."""
        url = f"{self._api_base}/alphas/{alpha_id}"
        payload = {"stage": "PROD"}
        resp = self._request_with_retry("PATCH", url, json=payload, timeout=30)
        return resp.json()

    # ── Batch simulation ──────────────────────────────────────────────────

    def batch_simulate(
        self,
        candidates: list[AlphaCandidate],
        *,
        max_concurrent: int = 3,
        timeout: float = 300.0,
    ) -> list[AlphaCandidate]:
        """Simulate multiple alphas concurrently. Mutates sim_result in-place.

        Uses the session-level _global_sem so multiple callers sharing the same
        session (e.g. MultiRegionRunner) never exceed the total concurrency cap.
        The max_concurrent parameter is kept for backward compatibility but the
        actual limit is set at construction time via WQSession(max_concurrent=N).
        """
        def _run(c: AlphaCandidate) -> AlphaCandidate:
            with self._global_sem:
                time.sleep(1.0)  # respect API rate limit
                try:
                    c.sim_result = self.simulate_and_parse(c.expr, c.settings, timeout=timeout)
                except Exception as exc:
                    logger.warning("Simulation failed for %s: %s", c.candidate_id, exc)
                    c.sim_result = SimulationResult(status="ERROR", error=str(exc))
            return c

        n_workers = min(len(candidates), self._max_concurrent) or 1
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run, c): c for c in candidates}
            results = []
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    logger.error("Unexpected error in batch_simulate: %s", exc)
        return results


def session_from_env() -> WQSession:
    """Build WQSession from environment variables (WQ_EMAIL / WQ_PASSWORD)."""
    email = os.environ.get("WQ_EMAIL", "")
    password = os.environ.get("WQ_PASSWORD", "")
    api_base = os.environ.get("WQ_API_BASE", WQ_API_BASE)
    max_concurrent = int(os.environ.get("WQ_MAX_CONCURRENT", "3"))
    if not email or not password:
        raise EnvironmentError(
            "WQ_EMAIL and WQ_PASSWORD must be set (in .env or environment)"
        )
    return WQSession(email, password, api_base=api_base, max_concurrent=max_concurrent)
