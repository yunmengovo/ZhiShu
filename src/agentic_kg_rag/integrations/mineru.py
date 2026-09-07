from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import httpx

from agentic_kg_rag.runtime_config import RUNTIME


class MinerUError(RuntimeError):
    status_code = 502


class MinerUService:
    def __init__(self, *, token: str, base_url: str) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")

    @property
    def configured(self) -> bool:
        return bool(self.token)

    def extract_markdown(self, *, filename: str, content: bytes) -> str:
        if not self.configured:
            raise MinerUError("MinerU 未配置；普通文档仍可使用内置解析器")
        headers = {"Authorization": f"Bearer {self.token}"}
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.post(
                f"{self.base_url}/api/v4/file-urls/batch",
                headers=headers,
                json={
                    "files": [{"name": Path(filename).name}],
                    "model_version": RUNTIME.mineru.model_version,
                    "enable_table": True,
                    "enable_formula": True,
                },
            )
            payload = self._payload(response, "申请上传地址")
            data = payload.get("data") or {}
            batch_id = str(data.get("batch_id", ""))
            upload_urls = data.get("file_urls") or []
            if not batch_id or not upload_urls:
                raise MinerUError("MinerU 未返回有效的批次或上传地址")
            upload = client.put(str(upload_urls[0]), content=content)
            if upload.status_code not in {200, 201, 204}:
                raise MinerUError(f"MinerU 文件上传失败：HTTP {upload.status_code}")

            result_url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
            deadline = time.monotonic() + RUNTIME.mineru.timeout_seconds
            zip_url = ""
            while time.monotonic() < deadline:
                result_payload = self._payload(
                    client.get(result_url, headers=headers),
                    "查询解析结果",
                )
                results = (result_payload.get("data") or {}).get("extract_result") or []
                if results:
                    result = results[0]
                    state = str(result.get("state", ""))
                    if state == "done":
                        zip_url = str(result.get("full_zip_url", ""))
                        break
                    if state == "failed":
                        raise MinerUError(
                            "MinerU 解析失败：" + str(result.get("err_msg") or "未知错误")
                        )
                time.sleep(RUNTIME.mineru.poll_interval_seconds)
            if not zip_url:
                raise MinerUError("MinerU 解析超时")
            archive = client.get(zip_url)
            archive.raise_for_status()
        return self._markdown_from_zip(archive.content)

    @staticmethod
    def _payload(response: httpx.Response, operation: str) -> dict[str, object]:
        try:
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            raise MinerUError(f"MinerU {operation}失败：HTTP {response.status_code}") from error
        if payload.get("code") != 0:
            raise MinerUError(f"MinerU {operation}失败：{payload.get('msg', '未知错误')}")
        return payload

    @staticmethod
    def _markdown_from_zip(content: bytes) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                candidates = [
                    name
                    for name in archive.namelist()
                    if name.casefold().endswith("full.md")
                ]
                if not candidates:
                    candidates = [
                        name for name in archive.namelist() if name.casefold().endswith(".md")
                    ]
                if not candidates:
                    raise MinerUError("MinerU 结果中没有 Markdown 文件")
                markdown = archive.read(min(candidates, key=len)).decode(
                    "utf-8",
                    errors="replace",
                )
        except zipfile.BadZipFile as error:
            raise MinerUError("MinerU 返回了无效的结果压缩包") from error
        if not markdown.strip():
            raise MinerUError("MinerU 返回的 Markdown 为空")
        return markdown
