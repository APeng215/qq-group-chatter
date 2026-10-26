from __future__ import annotations

import asyncio
import hashlib
import http.client
import ipaddress
import socket
import time
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urljoin

from qq_group_chatter.models import (
    ChatMessage, ConversationContext, ImageAttachment, ImageSnapshot, VisualImage,
)
from qq_group_chatter.observability import (
    IMAGE_DOWNLOAD_LATENCY_SECONDS, IMAGE_DOWNLOAD_TOTAL, log_event, record_error,
)


class ImageReadError(ValueError):
    pass


def image_media_type(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    raise ImageReadError("unsupported_format")


def validate_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ImageReadError("invalid_url")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        if parsed.hostname.lower() == "localhost":
            raise ImageReadError("private_address")
    else:
        if not address.is_global:
            raise ImageReadError("private_address")


def public_connection(address, timeout=8, source_address=None):
    """Resolve once and connect to that exact public address, retaining TLS hostname checks."""
    addresses = socket.getaddrinfo(*address, type=socket.SOCK_STREAM)
    if not addresses:
        raise ImageReadError("dns_failed")
    if any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ImageReadError("private_address")
    last_error = None
    for family, kind, proto, _, resolved in addresses:
        connection = socket.socket(family, kind, proto)
        try:
            connection.settimeout(timeout)
            connection.connect(resolved)
            return connection
        except OSError as exc:
            connection.close()
            last_error = exc
    raise last_error or ImageReadError("connect_failed")


class ImageUnderstandingService:
    def __init__(self, *, path: str | Path = ".mem0/images", enabled: bool = True,
                 max_images: int = 2, max_per_message: int = 8,
                 max_bytes: int = 8 * 1024 * 1024, timeout: float = 8, concurrency: int = 3,
                 downloader=None):
        self.path = Path(path)
        self.enabled = enabled
        self.max_images = max(1, max_images)
        self.max_per_message = max(1, max_per_message)
        self.max_bytes = max(1, max_bytes)
        self.timeout = max(0.1, timeout)
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._downloader = downloader
        self._snapshots: weakref.WeakValueDictionary[int, ImageSnapshot] = weakref.WeakValueDictionary()
        self._prepared: weakref.WeakValueDictionary[int, ImageAttachment] = weakref.WeakValueDictionary()
        self._references: Callable[[], set[str]] = lambda: set()
        self._worker: asyncio.Task | None = None

    async def start(self, references: Callable[[], set[str]]) -> None:
        self._references = references
        if self.enabled and self._worker is None:
            self._worker = asyncio.create_task(self._cleanup_worker())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        if self.enabled:
            self.collect_unused()

    async def _cleanup_worker(self):
        while True:
            await asyncio.sleep(60)
            try:
                self.collect_unused()
            except Exception as exc:
                record_error("image_cleanup", exc)

    def settings(self) -> dict[str, object]:
        return {"enabled": self.enabled, "max_images_per_request": self.max_images,
                "max_images_per_message": self.max_per_message}

    async def _download(self, url: str) -> bytes:
        return await asyncio.to_thread(self._download_sync, url)

    def _download_sync(self, url: str) -> bytes:
        deadline = time.monotonic() + self.timeout
        for _ in range(6):
            validate_public_url(url)
            parsed = urlsplit(url)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
            connection = connection_type(parsed.hostname, parsed.port, timeout=remaining)
            connection._create_connection = public_connection
            try:
                target = parsed.path or "/"
                if parsed.query:
                    target += "?" + parsed.query
                connection.request("GET", target, headers={"Accept": "image/*"})
                response = connection.getresponse()
                if response.status in {301, 302, 303, 307, 308}:
                    url = urljoin(url, response.getheader("Location", ""))
                    continue
                if response.status != 200:
                    raise ImageReadError("http_error")
                length = response.getheader("Content-Length")
                if length and int(length) > self.max_bytes:
                    raise ImageReadError("too_large")
                data = bytearray()
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError()
                    if connection.sock:
                        connection.sock.settimeout(remaining)
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    data.extend(chunk)
                    if len(data) > self.max_bytes:
                        raise ImageReadError("too_large")
                return bytes(data)
            finally:
                connection.close()
        raise ImageReadError("too_many_redirects")

    async def prepare(self, attachments: list[ImageAttachment]) -> list[ImageAttachment]:
        async def read(attachment: ImageAttachment, index: int):
            if not self.enabled:
                return replace(attachment, status="disabled", source_url=None)
            if index >= self.max_per_message:
                return replace(attachment, status="message_limit", source_url=None)
            if not attachment.source_url:
                return replace(attachment, status="missing_url")
            start = time.perf_counter()
            status = "ready"
            try:
                async with self._semaphore:
                    validate_public_url(attachment.source_url)
                    async with asyncio.timeout(self.timeout):
                        data = await (self._downloader or self._download)(attachment.source_url)
                    if len(data) > self.max_bytes:
                        raise ImageReadError("too_large")
                    media_type = image_media_type(data)
                    resource_id = hashlib.sha256(data).hexdigest()
                    self.path.mkdir(parents=True, exist_ok=True)
                    target = self.path / resource_id
                    # No await between write and return; grace period protects handoff to short-term storage.
                    target.write_bytes(data)
                prepared = replace(attachment, source_url=None, status="ready", resource_id=resource_id,
                                   media_type=media_type, size_bytes=len(data),
                                   duration_ms=round((time.perf_counter() - start) * 1000, 3))
                self._prepared[id(prepared)] = prepared
                return prepared
            except Exception as exc:
                status = str(exc) if isinstance(exc, ImageReadError) else (
                    "timeout" if isinstance(exc, TimeoutError) else "download_failed")
                record_error("image_download", ImageReadError(status))
                return replace(attachment, source_url=None, status=status,
                               duration_ms=round((time.perf_counter() - start) * 1000, 3))
            finally:
                duration = time.perf_counter() - start
                IMAGE_DOWNLOAD_TOTAL.labels(result=status).inc()
                IMAGE_DOWNLOAD_LATENCY_SECONDS.observe(duration)
                log_event("image_download", result=status, duration_ms=round(duration * 1000, 3))
        return list(await asyncio.gather(*(read(a, i) for i, a in enumerate(attachments))))

    def read_resource(self, resource_id: str) -> tuple[bytes, str] | None:
        if len(resource_id) != 64 or any(c not in "0123456789abcdef" for c in resource_id):
            return None
        target = self.path / resource_id
        if not target.is_file() or target.is_symlink():
            return None
        try:
            if target.stat().st_size > self.max_bytes:
                return None
            data = target.read_bytes()
            return data, image_media_type(data)
        except (OSError, ImageReadError):
            return None

    def snapshot(self, messages: list[ChatMessage], context: ConversationContext) -> ImageSnapshot:
        snapshot = ImageSnapshot(enabled=self.enabled, max_images=self.max_images)
        current = [m for m in messages if m.message_id == context.message_id]
        quoted = [m for m in messages if m.message_id != context.message_id
                  and m.message_id == context.reply_to_message_id]
        history = [m for m in reversed(messages) if m not in current and m not in quoted]
        selected: dict[str, int] = {}
        for message in [*current, *quoted, *history]:
            for attachment in message.attachments:
                entry: dict[str, object] = {
                    "message_id": message.message_id, "user_id": message.user_id,
                    "nickname": message.nickname, "position": attachment.position,
                    "is_current": message.message_id == context.message_id,
                    "resource_id": attachment.resource_id, "status": attachment.status,
                    "media_type": attachment.media_type, "size_bytes": attachment.size_bytes,
                    "duration_ms": attachment.duration_ms,
                }
                resource_id = attachment.resource_id
                if not self.enabled:
                    entry["status"] = "disabled"
                elif resource_id in selected:
                    entry.update(status="sent_duplicate", input_index=selected[resource_id])
                elif attachment.status == "ready" and resource_id:
                    if len(snapshot.images) >= self.max_images:
                        entry["status"] = "request_limit"
                    else:
                        resource = self.read_resource(resource_id)
                        if resource is None:
                            entry["status"] = "unavailable"
                        else:
                            data, media_type = resource
                            snapshot.images.append(VisualImage(resource_id, media_type, data))
                            selected[resource_id] = len(snapshot.images)
                            entry.update(status="sent", input_index=len(snapshot.images))
                snapshot.attachments.append(entry)
        self._snapshots[id(snapshot)] = snapshot
        return snapshot

    def collect_unused(self, *, grace_seconds: float = 60) -> None:
        keep = self._references()
        keep.update(a.resource_id for a in list(self._prepared.values()) if a.resource_id)
        for snapshot in list(self._snapshots.values()):
            keep.update(image.resource_id for image in snapshot.images)
        if not self.path.exists():
            return
        for target in self.path.iterdir():
            if (target.name not in keep and len(target.name) == 64 and target.is_file()
                    and not target.is_symlink() and time.time() - target.stat().st_mtime >= grace_seconds):
                target.unlink(missing_ok=True)
