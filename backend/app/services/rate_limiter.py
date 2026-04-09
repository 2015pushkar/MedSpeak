from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx


@dataclass
class RateLimitStatus:
    remaining: int
    daily_limit: int
    reset_at: datetime
    limit_exceeded: bool


class InMemoryRateLimitStore:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    async def get(self, key: str) -> int:
        return self._counts.get(key, 0)

    async def increment(self, key: str) -> int:
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    async def ping(self) -> bool:
        return True


class UpstashRestStore:
    def __init__(self, rest_url: str, rest_token: str) -> None:
        self.rest_url = rest_url.rstrip("/")
        self.rest_token = rest_token

    async def get(self, key: str) -> int:
        result = await self._command("get", key)
        if result in (None, "null"):
            return 0
        return int(result)

    async def increment(self, key: str) -> int:
        result = await self._command("incr", key)
        return int(result)

    async def expire(self, key: str, ttl_seconds: int) -> None:
        await self._command("expire", key, str(ttl_seconds))

    async def ping(self) -> bool:
        result = await self._command("ping")
        return str(result).upper() == "PONG"

    async def _command(self, *parts: str):
        url = "/".join([self.rest_url, *parts])
        headers = {"Authorization": f"Bearer {self.rest_token}"}
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()
        return payload.get("result")


class RateLimiter:
    def __init__(
        self,
        *,
        daily_limit: int,
        rest_url: str | None = None,
        rest_token: str | None = None,
    ) -> None:
        self.daily_limit = daily_limit
        if rest_url and rest_token:
            self._store = UpstashRestStore(rest_url, rest_token)
            self.backend = "upstash"
        else:
            self._store = InMemoryRateLimitStore()
            self.backend = "memory"

    async def peek(self, client_ip: str) -> RateLimitStatus:
        key, reset_at, _ = self._bucket(client_ip)
        count = await self._store.get(key)
        return self._status_from_count(count, reset_at)

    async def consume(self, client_ip: str) -> RateLimitStatus:
        key, reset_at, ttl_seconds = self._bucket(client_ip)
        count = await self._store.increment(key)
        if hasattr(self._store, "expire") and count == 1:
            await self._store.expire(key, ttl_seconds)
        return self._status_from_count(count, reset_at)

    async def healthcheck(self) -> dict[str, str]:
        try:
            healthy = await self._store.ping()
        except Exception as exc:  # pragma: no cover
            return {"status": "degraded", "backend": self.backend, "error": str(exc)}
        return {"status": "ok" if healthy else "degraded", "backend": self.backend}

    def _bucket(self, client_ip: str) -> tuple[str, datetime, int]:
        now = datetime.now(UTC)
        tomorrow = (now + timedelta(days=1)).date()
        reset_at = datetime.combine(tomorrow, datetime.min.time(), tzinfo=UTC)
        ttl_seconds = max(int((reset_at - now).total_seconds()), 1)
        bucket_date = now.date().isoformat()
        safe_ip = client_ip.replace(":", "_")
        return f"rate-limit:{bucket_date}:{safe_ip}", reset_at, ttl_seconds

    def _status_from_count(self, count: int, reset_at: datetime) -> RateLimitStatus:
        remaining = max(self.daily_limit - min(count, self.daily_limit), 0)
        return RateLimitStatus(
            remaining=remaining,
            daily_limit=self.daily_limit,
            reset_at=reset_at,
            limit_exceeded=count > self.daily_limit,
        )
