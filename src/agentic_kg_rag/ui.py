from __future__ import annotations

import html
import math
import mimetypes
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

from agentic_kg_rag.config import get_settings

DOCUMENT_HEADERS = [
    "标题",
    "文件名",
    "类型",
    "分块数",
    "向量状态",
    "实体 / 关系",
    "解析 / 构建状态",
    "文档 ID",
]
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".md", ".markdown", ".mdx", ".txt", ".html", ".htm"]
AGENTIC_MODE = "知枢智能模式"

CSS = """
:root {
  --zs-ink: #17233b;
  --zs-muted: #64748b;
  --zs-blue: #2563eb;
  --zs-cyan: #0891b2;
  --zs-teal: #0f766e;
  --zs-bg: #f4f7fb;
  --zs-line: #dce5f0;
  --zs-card: #ffffff;
  --zs-font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB",
    "Microsoft YaHei", sans-serif;
  --zs-font-mono: "SFMono-Regular", Menlo, Monaco, Consolas,
    "Liberation Mono", monospace;
  --font: var(--zs-font-sans);
  --font-mono: var(--zs-font-mono);
}
body, .gradio-container { background: var(--zs-bg) !important; }
.gradio-container {
  width: 100% !important;
  max-width: none !important;
  margin: 0 !important;
  padding: 22px 28px 40px 242px !important;
  color: var(--zs-ink) !important;
  font-family: var(--zs-font-sans) !important;
  font-variant-ligatures: common-ligatures;
  text-rendering: optimizeLegibility;
}
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .prose {
  font-family: var(--zs-font-sans) !important;
}
.gradio-container code,
.gradio-container pre,
.gradio-container kbd,
.gradio-container samp {
  font-family: var(--zs-font-mono) !important;
}
.brand-hero {
  position: relative;
  display: flex;
  align-items: center;
  gap: 20px;
  min-height: 126px;
  overflow: hidden;
  padding: 24px 30px;
  border-radius: 24px;
  color: #fff;
  background:
    radial-gradient(circle at 88% 18%, rgba(34,211,238,.26), transparent 26%),
    linear-gradient(128deg, #10233f 0%, #124b67 56%, #0f766e 100%);
  box-shadow: 0 18px 44px rgba(15, 47, 79, .18);
}
.brand-hero::after {
  content: "";
  position: absolute;
  right: 5%; top: -90px;
  width: 260px; height: 260px;
  border: 1px solid rgba(255,255,255,.14);
  border-radius: 50%;
  box-shadow: 0 0 0 38px rgba(255,255,255,.045), 0 0 0 78px rgba(255,255,255,.025);
}
.brand-mark {
  display: grid;
  place-items: center;
  flex: 0 0 68px;
  height: 68px;
  border: 1px solid rgba(255,255,255,.34);
  border-radius: 20px;
  background: rgba(255,255,255,.13);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.2);
  font-size: 34px;
  font-weight: 750;
}
.brand-copy { position: relative; z-index: 1; flex: 1; }
.brand-eyebrow {
  margin-bottom: 4px;
  color: #a5f3fc;
  font-size: 11px;
  font-weight: 750;
  letter-spacing: .17em;
}
.brand-copy h1 { margin: 0; color: #fff; font-size: 32px; line-height: 1.1; }
.brand-copy h1 span { margin-left: 8px; color: #b6e7ed; font-size: 20px; font-weight: 500; }
.brand-copy p { margin: 8px 0 0; color: #dbeafe; font-size: 14px; }
.brand-pills { position: relative; z-index: 1; display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.brand-pills span {
  padding: 7px 11px;
  border: 1px solid rgba(255,255,255,.2);
  border-radius: 999px;
  background: rgba(15,23,42,.2);
  color: #e6fbff;
  font-size: 12px;
}
.section-bar { align-items: center !important; margin: 20px 2px 10px !important; }
.section-title h2 { margin: 0; color: var(--zs-ink); font-size: 18px; }
.section-title p { margin: 3px 0 0; color: var(--zs-muted); font-size: 12px; }
.status-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
.status-card {
  min-height: 92px;
  padding: 15px 16px;
  border: 1px solid var(--zs-line);
  border-radius: 17px;
  background: var(--zs-card);
  box-shadow: 0 6px 20px rgba(30, 64, 100, .06);
}
.status-card .label { display: flex; align-items: center; gap: 7px; color: var(--zs-muted); font-size: 12px; }
.status-card .dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 0 4px #d1fae5; }
.status-card .dot.warn { background: #f59e0b; box-shadow: 0 0 0 4px #fef3c7; }
.status-card strong { display: block; margin-top: 11px; color: var(--zs-ink); font-size: 19px; line-height: 1.1; }
.status-card small { display: block; margin-top: 5px; color: var(--zs-muted); font-size: 11px; }
.status-error { padding: 16px; border: 1px solid #fecaca; border-radius: 16px; background: #fff1f2; color: #b91c1c; }
.app-shell { display: block !important; width: 100% !important; }
.side-nav {
  position: fixed !important;
  z-index: 20 !important;
  top: 22px !important;
  bottom: 22px !important;
  left: 22px !important;
  width: 196px !important;
  min-width: 196px !important;
  max-width: 196px !important;
  min-height: calc(100vh - 44px) !important;
  align-content: flex-start !important;
  gap: 8px !important;
  padding: 16px 14px !important;
  border: 1px solid var(--zs-line) !important;
  border-radius: 22px !important;
  background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%) !important;
  box-shadow: 0 12px 34px rgba(30, 64, 100, .12) !important;
}
.side-brand { display: flex; align-items: center; gap: 10px; padding: 3px 6px 16px; border-bottom: 1px solid #e7edf5; }
.side-brand-mark {
  display: grid; place-items: center; flex: 0 0 38px; height: 38px;
  border-radius: 12px; color: #fff; background: linear-gradient(135deg, #17345a, #0891b2);
  box-shadow: 0 7px 16px rgba(8, 145, 178, .18); font-size: 20px; font-weight: 760;
}
.side-brand-copy strong { display: block; color: var(--zs-ink); font-size: 15px; line-height: 1.1; }
.side-brand-copy span { display: block; margin-top: 4px; color: var(--zs-muted); font-size: 9px; letter-spacing: .13em; }
.side-nav-title { padding: 15px 7px 7px; }
.side-nav-title strong { display: block; color: var(--zs-ink); font-size: 12px; }
.side-nav-title span { display: block; margin-top: 3px; color: var(--zs-muted); font-size: 9px; }
.side-nav-control, .side-nav-control > div { width: 100% !important; }
.side-nav-control .wrap {
  display: flex !important;
  flex-direction: column !important;
  gap: 7px !important;
}
.side-nav-control label {
  position: relative !important;
  display: flex !important;
  align-items: center !important;
  width: 100% !important;
  min-height: 46px !important;
  padding: 10px 12px !important;
  border: 0 !important;
  border-radius: 12px !important;
  color: #475569 !important;
  background: transparent !important;
  font-weight: 650 !important;
  cursor: pointer !important;
}
.side-nav-control label::before {
  content: "•";
  display: inline-grid;
  place-items: center;
  width: 26px;
  height: 26px;
  margin-right: 9px;
  border-radius: 8px;
  background: #edf4fb;
  color: #19708b;
  font-size: 13px;
}
.side-nav-control label:nth-of-type(1)::before { content: "库"; }
.side-nav-control label:nth-of-type(2)::before { content: "问"; }
.side-nav-control label:hover { background: #f1f6fb !important; }
.side-nav-control label:has(input:checked) {
  color: #fff !important;
  background: linear-gradient(135deg, #2563eb, #0891b2) !important;
  box-shadow: 0 7px 16px rgba(37, 99, 235, .2) !important;
}
.side-nav-control label:has(input:checked)::before {
  color: #fff;
  background: rgba(255,255,255,.16);
}
.side-nav-control input { position: absolute !important; opacity: 0 !important; pointer-events: none !important; }
.workspace-host {
  width: 100% !important;
  min-width: 0 !important;
  max-width: none !important;
  gap: 0 !important;
}
.workspace-grid { align-items: flex-start !important; gap: 16px !important; padding-top: 0 !important; }
.left-workspace, .right-workspace { gap: 16px !important; min-width: 0 !important; }
.surface-card {
  gap: 12px !important;
  padding: 17px !important;
  border: 1px solid var(--zs-line) !important;
  border-radius: 19px !important;
  background: var(--zs-card) !important;
  box-shadow: 0 7px 24px rgba(30, 64, 100, .065) !important;
}
.surface-card > .form { border: 0 !important; }
.qa-toolbar-grid, .qa-main-grid { align-items: stretch !important; gap: 16px !important; }
.qa-toolbar-card, .memory-card { min-height: 0 !important; height: 100% !important; }
.qa-main-column { gap: 16px !important; min-width: 0 !important; }
.card-heading { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1px; }
.card-heading .icon {
  display: inline-grid; place-items: center; width: 31px; height: 31px; margin-right: 9px;
  border-radius: 10px; background: #e0f2fe; color: #0369a1; font-weight: 700;
}
.card-heading strong { color: var(--zs-ink); font-size: 15px; }
.card-heading p { margin: 4px 0 0; color: var(--zs-muted); font-size: 11px; }
.conversation-actions { align-items: end !important; gap: 9px !important; }
.conversation-actions button { min-width: 108px !important; height: 42px !important; }
.danger-button {
  color: #b42318 !important;
  border-color: #fecaca !important;
  background: #fff5f5 !important;
}
.danger-button:hover { background: #fee2e2 !important; }
.chat-card { height: 100% !important; padding-bottom: 12px !important; }
#zhishu-chatbot { border: 1px solid #e5ebf3 !important; border-radius: 15px !important; overflow: hidden; }
.composer-card { min-height: 0 !important; }
.trace-card { min-height: 270px !important; }
.composer-actions { align-items: end !important; }
#zhishu-send { min-width: 112px !important; height: 74px !important; }
.memory-shell { color: var(--zs-ink); }
.memory-top { display: flex; align-items: flex-end; justify-content: space-between; margin: 14px 0 8px; }
.memory-top span { color: var(--zs-muted); font-size: 12px; }
.memory-top strong { font-size: 24px; }
.memory-top small { color: var(--zs-muted); font-size: 12px; }
.memory-track { height: 8px; overflow: hidden; border-radius: 99px; background: #e8eef5; }
.memory-track i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg,#2563eb,#06b6d4); }
.memory-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-top: 14px; }
.memory-stat { padding: 10px; border-radius: 12px; background: #f5f8fc; }
.memory-stat span { display: block; color: var(--zs-muted); font-size: 10px; }
.memory-stat strong { display: block; margin-top: 4px; font-size: 13px; }
.memory-note { margin-top: 12px; padding: 9px 11px; border-radius: 11px; background: #ecfeff; color: #155e75; font-size: 11px; }
.memory-warning { margin-top: 10px; color: #b45309; font-size: 11px; }
.evidence-panel { min-height: 170px; max-height: 560px; overflow: auto; padding-right: 5px; }
.evidence-panel h3 { color: var(--zs-ink); border-bottom: 1px solid #e8eef5; padding-bottom: 9px; }
.library-table { overflow: hidden; }
.manager-row { align-items: end !important; gap: 10px !important; }
.manager-row button { min-width: 96px !important; }
.graph-canvas {
  min-height: 420px;
  overflow: hidden;
  border: 1px solid #dce8f3;
  border-radius: 16px;
  background: radial-gradient(circle at 50% 48%, #f8fdff 0%, #f3f7fb 70%);
}
.graph-canvas svg { display: block; width: 100%; height: auto; min-height: 420px; }
.graph-empty { display: grid; place-items: center; min-height: 300px; color: var(--zs-muted); }
.job-list { display: grid; gap: 8px; }
.job-row { padding: 10px 12px; border-radius: 12px; background: #f5f8fc; }
.job-row strong { color: var(--zs-ink); font-size: 12px; }
.job-row span { float: right; color: var(--zs-cyan); font-size: 11px; }
.job-row small { display: block; margin-top: 5px; color: var(--zs-muted); }
.job-track { height: 5px; margin-top: 7px; overflow: hidden; border-radius: 99px; background: #e2e8f0; }
.job-track i { display: block; height: 100%; background: linear-gradient(90deg,#2563eb,#06b6d4); }
.hint-strip { padding: 10px 12px; border-radius: 12px; background: #f0f9ff; color: #075985; font-size: 12px; }
button.primary { background: linear-gradient(135deg, var(--zs-blue), var(--zs-cyan)) !important; border: 0 !important; }
footer { display: none !important; }
@media (max-width: 1050px) {
  .gradio-container { padding: 14px !important; }
  .status-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .brand-pills { display: none; }
  .app-shell { display: flex !important; flex-direction: column !important; }
  .side-nav {
    position: static !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: none !important;
    min-height: 0 !important;
    padding: 10px !important;
  }
  .workspace-host { width: 100% !important; }
  .qa-toolbar-grid, .qa-main-grid { flex-direction: column !important; }
  .side-brand, .side-nav-title { display: none; }
  .side-nav-control .wrap { flex-direction: row !important; }
  .side-nav-control label { justify-content: center !important; }
}
@media (max-width: 700px) {
  .gradio-container { padding: 12px !important; }
  .brand-hero { padding: 20px; border-radius: 18px; }
  .brand-mark { flex-basis: 54px; height: 54px; border-radius: 15px; font-size: 27px; }
  .brand-copy h1 { font-size: 26px; }
  .status-grid, .memory-grid { grid-template-columns: 1fr; }
}
"""
THEME = gr.themes.Soft(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate")


class BackendError(RuntimeError):
    pass


class BackendClient:
    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                response = client.request(method, path, **kwargs)
            response.raise_for_status()
            if response.status_code == 204:
                return None
            return response.json()
        except httpx.HTTPStatusError as error:
            try:
                detail = error.response.json().get("detail", error.response.text)
            except ValueError:
                detail = error.response.text
            raise BackendError(f"后端请求失败（{error.response.status_code}）：{detail}") from error
        except httpx.TimeoutException as error:
            raise BackendError(
                "后端请求超时，服务可能仍在处理。请先查看任务状态，勿重复提交。"
            ) from error
        except httpx.HTTPError as error:
            raise BackendError(f"无法连接 API：{self.base_url}，请检查后端是否运行") from error

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def rag_status(self) -> dict[str, Any]:
        return self._request("GET", "/rag/status")

    def graph_status(self) -> dict[str, Any]:
        return self._request("GET", "/graph/status")

    def integration_status(self) -> dict[str, Any]:
        return self._request("GET", "/integrations/status")

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        return self._request("GET", "/knowledge-bases")

    def create_knowledge_base(self, name: str, description: str = "") -> dict[str, Any]:
        return self._request(
            "POST",
            "/knowledge-bases",
            json={"name": name, "description": description},
        )

    def update_knowledge_base(
        self,
        knowledge_base_id: str,
        *,
        name: str,
    ) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/knowledge-bases/{knowledge_base_id}",
            json={"name": name},
        )

    def delete_knowledge_base(self, knowledge_base_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/knowledge-bases/{knowledge_base_id}")

    def list_documents(self, knowledge_base_id: str | None = None) -> list[dict[str, Any]]:
        params = {"knowledge_base_id": knowledge_base_id} if knowledge_base_id else None
        return self._request("GET", "/documents", params=params)

    def list_assets(self, knowledge_base_id: str | None = None) -> list[dict[str, Any]]:
        params = {"knowledge_base_id": knowledge_base_id} if knowledge_base_id else None
        return self._request("GET", "/workspace/assets", params=params)

    def upload_document(
        self,
        path: str,
        *,
        knowledge_base_id: str = "kb-default",
        use_mineru: bool = False,
    ) -> dict[str, Any]:
        file_path = Path(path)
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        with file_path.open("rb") as stream:
            return self._request(
                "POST",
                "/integrations/mineru/ingest" if use_mineru else "/documents",
                files={"file": (file_path.name, stream, content_type)},
                data={"knowledge_base_id": knowledge_base_id},
            )

    def upload_documents_batch(
        self,
        paths: list[str],
        *,
        knowledge_base_id: str,
    ) -> dict[str, Any]:
        with ExitStack() as closing:
            uploads = []
            for path in paths:
                file_path = Path(path)
                stream = closing.enter_context(file_path.open("rb"))
                content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
                uploads.append(("files", (file_path.name, stream, content_type)))
            return self._request(
                "POST",
                "/documents/batch",
                files=uploads,
                data={"knowledge_base_id": knowledge_base_id},
            )

    def index_document(self, document_id: str) -> dict[str, Any]:
        return self._request("POST", f"/rag/index/{document_id}")

    def extract_graph(self, document_id: str) -> dict[str, Any]:
        return self._request("POST", f"/graph/extract/{document_id}")

    def delete_document(self, document_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/documents/{document_id}")

    def submit_document_rebuild(
        self,
        document_id: str,
        *,
        reparse: bool = True,
        rebuild_vector: bool = True,
        rebuild_graph: bool = True,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/jobs/documents/{document_id}/rebuild",
            json={
                "reparse": reparse,
                "rebuild_vector": rebuild_vector,
                "rebuild_graph": rebuild_graph,
            },
        )

    def list_jobs(self) -> list[dict[str, Any]]:
        return self._request("GET", "/jobs", params={"limit": 20})

    def graph_subgraph(
        self,
        *,
        query: str = "",
        document_ids: list[str] | None = None,
        limit: int = 80,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/graph/subgraph",
            json={"query": query, "document_ids": document_ids or [], "limit": limit},
        )

    def ask(
        self,
        question: str,
        *,
        top_k: int,
        document_ids: list[str],
        knowledge_base_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/rag/ask",
            json={
                "question": question,
                "top_k": top_k,
                "document_ids": document_ids,
                "knowledge_base_id": knowledge_base_id,
            },
        )

    def agent_ask(
        self,
        question: str,
        *,
        top_k: int,
        document_ids: list[str],
        session_id: str | None = None,
        knowledge_base_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/agent/ask",
            json={
                "question": question,
                "top_k": top_k,
                "document_ids": document_ids,
                "session_id": session_id,
                "knowledge_base_id": knowledge_base_id,
            },
        )

    def create_conversation(self, title: str = "新会话") -> dict[str, Any]:
        return self._request("POST", "/memory/conversations", json={"title": title})

    def list_conversations(self) -> list[dict[str, Any]]:
        return self._request("GET", "/memory/conversations")

    def get_conversation(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/memory/conversations/{session_id}")

    def delete_conversation(self, session_id: str) -> None:
        self._request("DELETE", f"/memory/conversations/{session_id}")


def _document_choices(documents: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (f"{item['title']} · {item['filename']}", item["document_id"])
        for item in documents
        if item.get("chunk_count", 0) > 0
    ]


def _document_rows(documents: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            item["title"],
            item["filename"],
            item["file_type"],
            item["chunk_count"],
            f"{item.get('vector_status', 'unknown')} · {item.get('indexed_chunks', 0)}",
            f"{item.get('entity_count', 0)} / {item.get('relationship_count', 0)}",
            item["status"]
            + " / "
            + {
                "queued": "排队中",
                "running": "构建中",
                "failed": "构建失败",
                "succeeded": "已完成",
            }.get(item.get("build_status", ""), "未构建"),
            item["document_id"],
        ]
        for item in documents
    ]


def _status_markdown(
    status: dict[str, Any],
    graph_status: dict[str, Any],
    document_count: int,
    knowledge_base_count: int = 1,
    integrations: dict[str, Any] | None = None,
) -> str:
    collection_state = "已就绪" if status["collection_ready"] else "未建立"
    chat_state = "正常" if status["chat_configured"] else "缺少配置"
    embedding_state = "正常" if status["embedding_configured"] else "缺少配置"
    graph_connected = graph_status.get("connected", True)
    graph_icon = "🟢" if graph_connected else "🟠"
    graph_backend = graph_status.get("backend", graph_status.get("mode", "unknown"))
    graph_dot = "" if graph_connected else " warn"
    integrations = integrations or {}
    optional_count = sum(
        bool(integrations.get(key))
        for key in ("mineru_configured", "tavily_configured", "langsmith_enabled")
    )
    return (
        '<div class="status-grid">'
        '<div class="status-card"><div class="label"><i class="dot"></i>系统服务</div>'
        "<strong>API 在线</strong><small>文档与问答接口可用</small></div>"
        '<div class="status-card"><div class="label"><i class="dot"></i>知识库</div>'
        f"<strong>{knowledge_base_count} 个空间</strong><small>{document_count} 份文档 · "
        f"{status['indexed_chunks']} 个向量</small></div>"
        '<div class="status-card"><div class="label"><i class="dot"></i>向量检索</div>'
        f"<strong>{html.escape(str(status['mode']))}</strong><small>Qdrant · {collection_state}</small></div>"
        f'<div class="status-card"><div class="label"><i class="dot{graph_dot}"></i>知识图谱</div>'
        f"<strong>{html.escape(str(graph_backend))}</strong><small>{graph_status['entity_count']} 个实体 · "
        f"{graph_status['relationship_count']} 条关系 {graph_icon}</small></div>"
        '<div class="status-card"><div class="label"><i class="dot"></i>模型服务</div>'
        f"<strong>{chat_state} / {embedding_state}</strong><small>Chat / Embedding</small></div>"
        '<div class="status-card"><div class="label"><i class="dot"></i>可选增强</div>'
        f"<strong>{optional_count} / 3 已启用</strong><small>MinerU / Tavily / LangSmith</small></div>"
        "</div>"
    )


def _knowledge_base_choices(items: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (f"{item['name']} · {item.get('document_count', 0)} 份文档", item["knowledge_base_id"])
        for item in items
    ]


def _conversation_choices(conversations: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (
            f"{item['title']} · {item['message_count']} 条"
            + (
                f" · 已压缩 {item['compressed_messages']} 条" if item["compressed_messages"] else ""
            ),
            item["session_id"],
        )
        for item in conversations
    ]


def _conversation_history(conversation: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": item["role"], "content": item["content"]}
        for item in conversation.get("messages", [])
    ]


def render_memory_status(memory: dict[str, Any] | None) -> str:
    if not memory:
        return (
            '<div class="memory-shell"><div class="card-heading"><div>'
            '<span class="icon">忆</span><strong>会话记忆</strong>'
            "<p>尚未建立会话</p></div></div></div>"
        )
    threshold = memory.get("max_tokens", 0)
    context_window = memory.get("context_window_tokens", threshold)
    reserved_tokens = max(0, context_window - threshold)
    percentage = min(100.0, (memory["token_count"] / threshold * 100) if threshold else 0)
    compressed_now = " · 本轮已压缩" if memory.get("compressed_this_turn") else ""
    warning = (
        f'<div class="memory-warning">⚠ {html.escape(str(memory["compression_warning"]))}</div>'
        if memory.get("compression_warning")
        else ""
    )
    return (
        '<div class="memory-shell"><div class="card-heading"><div>'
        '<span class="icon">忆</span><strong>会话记忆</strong>'
        "<p>长期保留关键信息，按预算自动滚动压缩</p></div></div>"
        '<div class="memory-top"><span>会话记忆用量</span>'
        f"<div><strong>{memory['token_count']:,}</strong><small> / {threshold:,} tokens</small></div></div>"
        f'<div class="memory-track"><i style="width:{percentage:.2f}%"></i></div>'
        '<div class="memory-grid">'
        f'<div class="memory-stat"><span>模型窗口</span><strong>{context_window:,}</strong></div>'
        f'<div class="memory-stat"><span>生成预留</span><strong>{reserved_tokens:,}</strong></div>'
        f'<div class="memory-stat"><span>保留消息</span><strong>{memory["message_count"]} 条</strong></div>'
        "</div>"
        f'<div class="memory-note">已压缩旧消息 {memory["compressed_messages"]} 条{compressed_now}</div>'
        f"{warning}</div>"
    )


def render_memory_notice(message: str, *, error: bool = False) -> str:
    tone = "memory-warning" if error else "memory-note"
    return (
        '<div class="memory-shell"><div class="card-heading"><div>'
        '<span class="icon">忆</span><strong>会话记忆</strong>'
        f"<p>{html.escape(message)}</p></div></div>"
        f'<div class="{tone}">{html.escape(message)}</div></div>'
    )


def initialize_conversations(
    client: BackendClient,
) -> tuple[Any, list[dict[str, str]], str]:
    try:
        conversations = client.list_conversations()
        if conversations:
            session_id = conversations[0]["session_id"]
            conversation = client.get_conversation(session_id)
        else:
            conversation = client.create_conversation()
            session_id = conversation["session_id"]
            conversations = [conversation]
        return (
            gr.update(choices=_conversation_choices(conversations), value=session_id),
            _conversation_history(conversation),
            render_memory_status(conversation),
        )
    except BackendError as error:
        return gr.update(choices=[], value=None), [], render_memory_notice(str(error), error=True)


def create_new_conversation(
    client: BackendClient,
) -> tuple[Any, list[dict[str, str]], str, str]:
    try:
        conversation = client.create_conversation()
        conversations = client.list_conversations()
        return (
            gr.update(
                choices=_conversation_choices(conversations),
                value=conversation["session_id"],
            ),
            [],
            render_memory_status(conversation),
            "### 执行详情\n新会话已创建，可以开始提问。",
        )
    except BackendError as error:
        return (
            gr.update(),
            [],
            render_memory_notice(str(error), error=True),
            f"### 执行详情\n🔴 {error}",
        )


def load_conversation(
    session_id: str | None,
    client: BackendClient,
) -> tuple[list[dict[str, str]], str, str]:
    if not session_id:
        return [], render_memory_notice("请选择会话。"), "### 执行详情\n等待提问。"
    try:
        conversation = client.get_conversation(session_id)
        return (
            _conversation_history(conversation),
            render_memory_status(conversation),
            "### 执行详情\n已恢复该会话的近期消息与滚动摘要。",
        )
    except BackendError as error:
        return [], render_memory_notice(str(error), error=True), f"### 执行详情\n🔴 {error}"


def delete_current_conversation(
    session_id: str | None,
    client: BackendClient,
) -> tuple[Any, list[dict[str, str]], str, str]:
    try:
        if session_id:
            client.delete_conversation(session_id)
        return (
            create_new_conversation(client)
            if not client.list_conversations()
            else _select_latest(client)
        )
    except BackendError as error:
        return (
            gr.update(),
            [],
            render_memory_notice(str(error), error=True),
            f"### 执行详情\n🔴 {error}",
        )


def _select_latest(client: BackendClient) -> tuple[Any, list[dict[str, str]], str, str]:
    conversations = client.list_conversations()
    conversation = client.get_conversation(conversations[0]["session_id"])
    return (
        gr.update(
            choices=_conversation_choices(conversations),
            value=conversation["session_id"],
        ),
        _conversation_history(conversation),
        render_memory_status(conversation),
        "### 执行详情\n会话已删除。",
    )


def render_jobs(jobs: list[dict[str, Any]]) -> str:
    if not jobs:
        return '<div class="job-list"><div class="job-row"><small>暂无后台任务</small></div></div>'
    labels = {
        "queued": "排队中",
        "running": "执行中",
        "succeeded": "已完成",
        "failed": "失败",
    }
    rows = ['<div class="job-list">']
    for job in jobs[:8]:
        progress = int(job.get("progress", 0))
        status = labels.get(job.get("status", ""), job.get("status", "unknown"))
        message = job.get("error") or job.get("message") or ""
        rows.append(
            '<div class="job-row">'
            f"<strong>{html.escape(str(job.get('job_type', 'task')))}</strong>"
            f"<span>{html.escape(str(status))} · {progress}%</span>"
            f"<small>{html.escape(str(message))}</small>"
            f'<div class="job-track"><i style="width:{progress}%"></i></div></div>'
        )
    rows.append("</div>")
    return "".join(rows)


def render_graph_visualization(graph: dict[str, Any]) -> str:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes or not edges:
        return (
            '<div class="graph-canvas"><div class="graph-empty">'
            "当前范围还没有可展示的关系，请先抽取知识图谱。"
            "</div></div>"
        )
    width, height = 960, 520
    center_x, center_y = width / 2, height / 2
    radius = min(width, height) * 0.37
    positions: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(nodes):
        ring = 1 if len(nodes) <= 24 or index < 24 else 0.62
        ring_items = min(len(nodes), 24) if ring == 1 else max(1, len(nodes) - 24)
        ring_index = index if ring == 1 else index - 24
        angle = (2 * math.pi * ring_index / ring_items) - math.pi / 2
        positions[node["node_id"]] = (
            center_x + radius * ring * math.cos(angle),
            center_y + radius * ring * math.sin(angle),
        )
    palette = ["#2563eb", "#0891b2", "#0f766e", "#7c3aed", "#c2410c", "#be185d"]
    svg = [
        (
            '<div class="graph-canvas"><svg viewBox="0 0 960 520" role="img" '
            'aria-label="知识图谱关系图">'
        ),
        (
            '<defs><marker id="zs-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker></defs>'
        ),
    ]
    for edge in edges:
        source = positions.get(edge["source_id"])
        target = positions.get(edge["target_id"])
        if not source or not target:
            continue
        x1, y1 = source
        x2, y2 = target
        label_x, label_y = (x1 + x2) / 2, (y1 + y2) / 2
        svg.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#b8c5d6" stroke-width="1.4" marker-end="url(#zs-arrow)"/>'
            f'<text x="{label_x:.1f}" y="{label_y - 5:.1f}" text-anchor="middle" '
            'font-size="9" fill="#64748b" paint-order="stroke" stroke="#f6fafc" '
            f'stroke-width="3">{html.escape(str(edge["relation"]))}</text>'
        )
    for node in nodes:
        x, y = positions[node["node_id"]]
        entity_type = str(node.get("entity_type", "Concept"))
        color = palette[sum(ord(char) for char in entity_type) % len(palette)]
        label = str(node["label"])
        short_label = label if len(label) <= 15 else label[:14] + "…"
        svg.append(
            f"<g><title>{html.escape(label)} · {html.escape(entity_type)}</title>"
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="22" fill="{color}" opacity=".94"/>'
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="9" '
            f'font-weight="700" fill="#fff">{html.escape(short_label[:7])}</text>'
            f'<text x="{x:.1f}" y="{y + 37:.1f}" text-anchor="middle" font-size="10" '
            f'fill="#334155">{html.escape(short_label)}</text></g>'
        )
    svg.append("</svg></div>")
    return "".join(svg)


def refresh_workspace(
    client: BackendClient,
    knowledge_base_id: str | None = None,
) -> tuple[str, Any, Any, list[list[Any]], Any, Any, str, str]:
    try:
        client.health()
        status = client.rag_status()
        graph_status = client.graph_status()
        integrations = client.integration_status()
        knowledge_bases = client.list_knowledge_bases()
        base_ids = {item["knowledge_base_id"] for item in knowledge_bases}
        selected_base = (
            knowledge_base_id
            if knowledge_base_id in base_ids
            else (knowledge_bases[0]["knowledge_base_id"] if knowledge_bases else None)
        )
        all_documents = client.list_documents()
        documents = client.list_assets(selected_base)
        choices = _document_choices(documents)
        document_ids = [item["document_id"] for item in documents]
        graph = client.graph_subgraph(document_ids=document_ids, limit=80)
        jobs = client.list_jobs()
        base_choices = _knowledge_base_choices(knowledge_bases)
        return (
            _status_markdown(
                status,
                graph_status,
                len(all_documents),
                len(knowledge_bases),
                integrations,
            ),
            gr.update(choices=base_choices, value=selected_base),
            gr.update(choices=base_choices, value=selected_base),
            _document_rows(documents),
            gr.update(choices=choices, value=[]),
            gr.update(choices=choices, value=[]),
            render_graph_visualization(graph),
            render_jobs(jobs),
        )
    except BackendError as error:
        message = f'<div class="status-error">{html.escape(str(error))}</div>'
        return (
            message,
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            [],
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            render_graph_visualization({}),
            render_jobs([]),
        )


def refresh_assets(
    knowledge_base_id: str | None,
    client: BackendClient,
) -> tuple[list[list[Any]], Any, str, str]:
    try:
        documents = client.list_assets(knowledge_base_id)
        choices = _document_choices(documents)
        graph = client.graph_subgraph(
            document_ids=[item["document_id"] for item in documents],
            limit=80,
        )
        return (
            _document_rows(documents),
            gr.update(choices=choices, value=[]),
            render_graph_visualization(graph),
            render_jobs(client.list_jobs()),
        )
    except BackendError:
        return [], gr.update(choices=[], value=[]), render_graph_visualization({}), render_jobs([])


def poll_build_status(
    knowledge_base_id: str | None,
    client: BackendClient,
    selected_ids: list[str] | None = None,
    query: str = "",
) -> tuple[Any, Any, Any]:
    try:
        if not knowledge_base_id:
            return gr.skip(), gr.skip(), gr.skip()
        assets = client.list_assets(knowledge_base_id)
        graph = (
            client.graph_subgraph(
                query=query,
                document_ids=selected_ids or [item["document_id"] for item in assets],
                limit=80,
            )
            if assets
            else {}
        )
        ids = {item["document_id"] for item in assets}
        jobs = [job for job in client.list_jobs() if job.get("target_id") in ids]
        return _document_rows(assets), render_jobs(jobs), render_graph_visualization(graph)
    except BackendError:
        return gr.skip(), gr.skip(), gr.skip()


def refresh_question_scope(
    knowledge_base_id: str | None,
    client: BackendClient,
) -> Any:
    try:
        return gr.update(
            choices=_document_choices(client.list_assets(knowledge_base_id)),
            value=[],
        )
    except BackendError:
        return gr.update(choices=[], value=[])


def upload_documents(
    files: list[str] | None,
    auto_index: bool,
    auto_graph: bool,
    use_mineru: bool,
    knowledge_base_id: str | None,
    client: BackendClient,
) -> tuple[str, str, list[list[Any]], Any, Any, str, str]:
    if not files:
        snapshot = refresh_workspace(client, knowledge_base_id)
        return (
            "请选择至少一个文档。",
            snapshot[0],
            snapshot[3],
            snapshot[4],
            snapshot[5],
            snapshot[6],
            snapshot[7],
        )
    if not knowledge_base_id:
        return (
            "请先选择知识空间。",
            '<div class="status-error">未选择知识空间</div>',
            [],
            gr.update(),
            gr.update(),
            render_graph_visualization({}),
            render_jobs([]),
        )

    results = []
    try:
        documents: list[dict[str, Any]] = []
        if use_mineru:
            for file_path in files:
                documents.append(
                    client.upload_document(
                        file_path,
                        knowledge_base_id=knowledge_base_id,
                        use_mineru=True,
                    )
                )
        else:
            batch = client.upload_documents_batch(
                files,
                knowledge_base_id=knowledge_base_id,
            )
            for item in batch.get("results", []):
                if item.get("success") and item.get("document"):
                    documents.append(item["document"])
                else:
                    results.append(
                        f"- `{item.get('filename', 'unknown')}`：🔴 {item.get('error', '导入失败')}"
                    )
        for document in documents:
            duplicate = "（已存在）" if document.get("duplicate") else ""
            message = (
                f"- `{document['filename']}`：解析为 {document['chunk_count']} 个分块{duplicate}"
            )
            if (auto_index or auto_graph) and document["chunk_count"] > 0:
                try:
                    job = client.submit_document_rebuild(
                        document["document_id"],
                        reparse=False,
                        rebuild_vector=auto_index,
                        rebuild_graph=auto_graph,
                    )
                    message += f"，构建任务 `{job['job_id']}` 已提交，进度将在任务面板更新"
                except BackendError as error:
                    message += f"，构建未提交：{error}（解析结果已保留）"
            results.append(message)
        operation = "### 批量导入完成\n" + "\n".join(results)
    except BackendError as error:
        operation = f"### 上传中断\n🔴 {error}\n\n" + "\n".join(results)

    snapshot = refresh_workspace(client, knowledge_base_id)
    return (
        operation,
        snapshot[0],
        snapshot[3],
        snapshot[4],
        snapshot[5],
        snapshot[6],
        snapshot[7],
    )


def index_selected(
    document_ids: list[str] | None,
    knowledge_base_id: str | None,
    client: BackendClient,
) -> tuple[str, str, list[list[Any]], str]:
    if not document_ids:
        snapshot = refresh_workspace(client, knowledge_base_id)
        return "请选择需要建立索引的文档。", snapshot[0], snapshot[3], snapshot[6]
    results = []
    for document_id in document_ids:
        try:
            job = client.submit_document_rebuild(
                document_id,
                reparse=False,
                rebuild_vector=True,
                rebuild_graph=False,
            )
            results.append(f"- `{document_id}`：向量索引任务 `{job['job_id']}` 已提交")
        except BackendError as error:
            results.append(f"- `{document_id}`：🔴 {error}")
    operation = "### 已提交后台索引\n" + "\n".join(results)
    snapshot = refresh_workspace(client, knowledge_base_id)
    return operation, snapshot[0], snapshot[3], snapshot[6]


def extract_selected_graph(
    document_ids: list[str] | None,
    knowledge_base_id: str | None,
    client: BackendClient,
) -> tuple[str, str, list[list[Any]], str]:
    if not document_ids:
        snapshot = refresh_workspace(client, knowledge_base_id)
        return "请选择需要抽取知识图谱的文档。", snapshot[0], snapshot[3], snapshot[6]
    results = []
    for document_id in document_ids:
        try:
            job = client.submit_document_rebuild(
                document_id,
                reparse=False,
                rebuild_vector=False,
                rebuild_graph=True,
            )
            results.append(f"- `{document_id}`：图谱抽取任务 `{job['job_id']}` 已提交")
        except BackendError as error:
            results.append(f"- `{document_id}`：🔴 {error}")
    operation = "### 已提交后台图谱抽取\n" + "\n".join(results)
    snapshot = refresh_workspace(client, knowledge_base_id)
    return operation, snapshot[0], snapshot[3], snapshot[6]


def create_knowledge_base(
    name: str,
    client: BackendClient,
) -> tuple[str, str, Any, Any, list[list[Any]], Any, Any, str, str]:
    try:
        created = client.create_knowledge_base(name.strip())
        message = f"### 知识空间已创建\n`{created['name']}`"
        snapshot = refresh_workspace(client, created["knowledge_base_id"])
    except BackendError as error:
        message = f"### 创建失败\n🔴 {error}"
        snapshot = refresh_workspace(client)
    return (message, *snapshot)


def rename_knowledge_base(
    knowledge_base_id: str | None,
    name: str,
    client: BackendClient,
) -> tuple[str, str, Any, Any, list[list[Any]], Any, Any, str, str]:
    if not knowledge_base_id or not name.strip():
        snapshot = refresh_workspace(client, knowledge_base_id)
        return ("请输入新的知识空间名称。", *snapshot)
    try:
        updated = client.update_knowledge_base(knowledge_base_id, name=name.strip())
        message = f"### 已重命名\n当前名称：`{updated['name']}`"
    except BackendError as error:
        message = f"### 重命名失败\n🔴 {error}"
    return (message, *refresh_workspace(client, knowledge_base_id))


def delete_knowledge_base(
    knowledge_base_id: str | None,
    confirmed: bool,
    client: BackendClient,
) -> tuple[str, str, Any, Any, list[list[Any]], Any, Any, str, str]:
    if not knowledge_base_id or not confirmed:
        snapshot = refresh_workspace(client, knowledge_base_id)
        return ("删除知识空间前请勾选确认。", *snapshot)
    try:
        result = client.delete_knowledge_base(knowledge_base_id)
        warnings = "\n".join(f"- ⚠ {item}" for item in result.get("warnings", []))
        message = f"### 知识空间已删除\n同步删除 {result['deleted_documents']} 份文档。" + (
            f"\n{warnings}" if warnings else ""
        )
    except BackendError as error:
        message = f"### 删除失败\n🔴 {error}"
    return (message, *refresh_workspace(client))


def delete_selected_documents(
    document_ids: list[str] | None,
    knowledge_base_id: str | None,
    confirmed: bool,
    client: BackendClient,
) -> tuple[str, str, list[list[Any]], Any, Any, str, str]:
    if not document_ids or not confirmed:
        snapshot = refresh_workspace(client, knowledge_base_id)
        return (
            "请选择文档并勾选删除确认。",
            snapshot[0],
            snapshot[3],
            snapshot[4],
            snapshot[5],
            snapshot[6],
            snapshot[7],
        )
    lines = []
    for document_id in document_ids:
        try:
            result = client.delete_document(document_id)
            lines.append(
                f"- `{result['filename']}`：文件、{result['deleted_vectors']} 个向量和图谱已清理"
            )
            lines.extend(f"  - ⚠ {warning}" for warning in result.get("warnings", []))
        except BackendError as error:
            lines.append(f"- `{document_id}`：🔴 {error}")
    snapshot = refresh_workspace(client, knowledge_base_id)
    return (
        "### 文档删除结果\n" + "\n".join(lines),
        snapshot[0],
        snapshot[3],
        snapshot[4],
        snapshot[5],
        snapshot[6],
        snapshot[7],
    )


def queue_selected_rebuilds(
    document_ids: list[str] | None,
    client: BackendClient,
) -> tuple[str, str]:
    if not document_ids:
        return "请选择需要完整重建的文档。", render_jobs(client.list_jobs())
    lines = []
    for document_id in document_ids:
        try:
            job = client.submit_document_rebuild(document_id)
            lines.append(f"- `{document_id}`：任务 `{job['job_id']}` 已进入队列")
        except BackendError as error:
            lines.append(f"- `{document_id}`：🔴 {error}")
    return "### 已提交后台任务\n" + "\n".join(lines), render_jobs(client.list_jobs())


def refresh_graph_view(
    document_ids: list[str] | None,
    knowledge_base_id: str | None,
    query: str,
    client: BackendClient,
) -> str:
    try:
        selected = document_ids or [
            item["document_id"] for item in client.list_assets(knowledge_base_id)
        ]
        return render_graph_visualization(
            client.graph_subgraph(query=query.strip(), document_ids=selected, limit=80)
        )
    except BackendError:
        return render_graph_visualization({})


def render_evidence(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "### 检索证据\n没有检索到可用证据。"
    cards = ["### 检索证据"]
    for item in evidence:
        location = []
        if item.get("section_title"):
            location.append(f"章节：{item['section_title']}")
        if item.get("page_start"):
            page = str(item["page_start"])
            if item.get("page_end") and item["page_end"] != item["page_start"]:
                page += f"–{item['page_end']}"
            location.append(f"页码：{page}")
        location_text = " · ".join(location) if location else "未标注章节/页码"
        excerpt = str(item["text"]).strip()
        if len(excerpt) > 900:
            excerpt = excerpt[:900].rstrip() + "…"
        cards.append(
            f"#### [{item['rank']}] {item['title']}\n"
            f"`{item['filename']}` · {location_text} · 相似度 `{item['score']:.4f}`\n\n"
            f"> {excerpt.replace(chr(10), chr(10) + '> ')}"
        )
    return "\n\n---\n\n".join(cards)


def render_agent_trace(result: dict[str, Any], mode: str) -> str:
    if mode != AGENTIC_MODE:
        return "### 执行轨迹\n`retrieve → generate_answer`（基础固定链路）"
    steps = result.get("steps", [])
    lines = [
        "### 执行轨迹",
        (
            f"路由：`{result.get('route', 'unknown')}` · "
            f"检索次数：`{result.get('retrieval_attempts', 0)}`"
        ),
    ]
    final_query = result.get("final_query")
    if final_query and final_query != result.get("question"):
        lines.append(f"最终查询：`{final_query}`")
    if result.get("used_web_search"):
        lines.append("联网补充：`Tavily`（仅本地证据不足且问题具有时效性时启用）")
    if result.get("citation_warnings"):
        lines.append("引用校验：⚠ " + "；".join(str(item) for item in result["citation_warnings"]))
    lines.extend(
        f"{index}. **{step['node']}** · `{step['action']}` — {step['detail']}"
        for index, step in enumerate(steps, 1)
    )
    return "\n\n".join(lines)


def render_graph_relations(relations: list[dict[str, Any]]) -> str:
    if not relations:
        return "### 图谱关系\n本次没有命中图谱关系。"
    lines = ["### 图谱关系"]
    lines.extend(
        f"- **[G{item['rank']}]** `{item['source']}` ({item['source_type']}) "
        f"— **{item['relation']}** → `{item['target']}` ({item['target_type']})"
        + (f"：{item['description']}" if item.get("description") else "")
        for item in relations
    )
    return "\n".join(lines)


def ask_question(
    message: str,
    history: list[dict[str, Any]] | None,
    document_ids: list[str] | None,
    top_k: int,
    mode: str,
    session_id: str | None,
    knowledge_base_id: str | None,
    client: BackendClient,
) -> tuple[list[dict[str, Any]], str, str, Any, str]:
    history = list(history or [])
    clean_message = message.strip()
    if not clean_message:
        return history, "", "### 执行详情\n请输入问题。", gr.update(), render_memory_status(None)
    history.append({"role": "user", "content": clean_message})
    try:
        if mode == AGENTIC_MODE:
            result = client.agent_ask(
                clean_message,
                top_k=int(top_k),
                document_ids=document_ids or [],
                session_id=session_id,
                knowledge_base_id=knowledge_base_id,
            )
        else:
            result = client.ask(
                clean_message,
                top_k=int(top_k),
                document_ids=document_ids or [],
                knowledge_base_id=knowledge_base_id,
            )
        history.append({"role": "assistant", "content": result["answer"]})
        panels = [render_agent_trace(result, mode), render_evidence(result["evidence"])]
        if mode == AGENTIC_MODE:
            panels.append(render_graph_relations(result.get("graph_relations", [])))
        evidence = "\n\n---\n\n".join(panels)
        resolved_session = result.get("session_id") or session_id
        conversations = client.list_conversations()
        conversation_update = gr.update(
            choices=_conversation_choices(conversations),
            value=resolved_session,
        )
        memory_status = (
            render_memory_status(result.get("memory"))
            if mode == AGENTIC_MODE
            else render_memory_notice("基础 RAG 用于基线对比，本轮不写入会话记忆。")
        )
    except BackendError as error:
        history.append({"role": "assistant", "content": f"请求失败：{error}"})
        evidence = f"### 执行详情\n🔴 {error}"
        conversation_update = gr.update()
        memory_status = render_memory_notice("本轮请求失败，未写入会话。", error=True)
    return history, "", evidence, conversation_update, memory_status


def build_ui(client: BackendClient | None = None) -> gr.Blocks:
    if client is None:
        settings = get_settings()
        client = BackendClient(settings.ui_api_base_url)

    with gr.Blocks(title="知枢 ZhiShu", fill_width=True) as demo:
        gr.HTML(
            """
            <header class="brand-hero">
              <div class="brand-mark">枢</div>
              <div class="brand-copy">
                <div class="brand-eyebrow">AGENTIC KNOWLEDGE NEXUS</div>
                <h1>知枢 <span>ZhiShu</span></h1>
                <p>让文档、知识关系与智能推理汇聚成可追溯的答案。</p>
              </div>
              <div class="brand-pills">
                <span>Vector RAG</span><span>Knowledge Graph</span><span>Agent Memory</span>
              </div>
            </header>
            """
        )
        with gr.Row(elem_classes="section-bar"):
            with gr.Column(scale=8):
                gr.HTML(
                    '<div class="section-title"><h2>运行概览</h2>'
                    "<p>知识、检索、模型与图谱服务状态</p></div>"
                )
            refresh_button = gr.Button("↻ 刷新状态", size="sm", scale=1)
        status_panel = gr.HTML('<div class="status-error">正在连接服务…</div>')

        with gr.Row(elem_classes="app-shell"):
            with gr.Column(elem_classes="side-nav"):
                gr.HTML(
                    '<div class="side-brand"><div class="side-brand-mark">枢</div>'
                    '<div class="side-brand-copy"><strong>知枢</strong><span>ZHISHU</span></div></div>'
                    '<div class="side-nav-title"><strong>工作台导航</strong>'
                    "<span>WORKSPACE</span></div>"
                )
                workspace_nav = gr.Radio(
                    choices=["知识空间", "智能问答"],
                    value="知识空间",
                    show_label=False,
                    container=False,
                    elem_classes="side-nav-control",
                )
            with gr.Column(elem_classes="workspace-host"):
                with gr.Column(visible=True) as knowledge_workspace:
                    with gr.Column(elem_classes=["surface-card", "manager-card"]):
                        gr.HTML(
                            '<div class="card-heading"><div><span class="icon">域</span>'
                            "<strong>知识空间管理</strong><p>不同空间的文档、检索范围与图谱互相隔离</p>"
                            "</div></div>"
                        )
                        with gr.Row(elem_classes="manager-row"):
                            knowledge_base_selector = gr.Dropdown(
                                choices=[],
                                label="当前知识空间",
                                scale=3,
                            )
                            knowledge_base_name = gr.Textbox(
                                label="名称",
                                placeholder="输入新名称或重命名",
                                scale=3,
                            )
                            create_base_button = gr.Button("新建", scale=1)
                            rename_base_button = gr.Button("重命名", scale=1)
                            delete_base_button = gr.Button(
                                "删除空间",
                                scale=1,
                                elem_classes="danger-button",
                            )
                        confirm_base_delete = gr.Checkbox(
                            value=False,
                            label="确认级联删除当前知识空间内的文档、向量与图谱",
                        )

                    with gr.Row(elem_classes="workspace-grid"):
                        with gr.Column(
                            scale=4,
                            elem_classes=["surface-card", "left-workspace"],
                        ):
                            gr.HTML(
                                '<div class="card-heading"><div><span class="icon">入</span>'
                                "<strong>批量导入资料</strong><p>支持 PDF、Word、Markdown、文本与 HTML</p>"
                                "</div></div>"
                            )
                            files = gr.File(
                                label="选择或拖入多份文档",
                                file_count="multiple",
                                file_types=SUPPORTED_FILE_TYPES,
                                type="filepath",
                                height=250,
                            )
                            gr.HTML(
                                '<div class="hint-strip">标准解析无需额外服务；复杂扫描 PDF 可选择 MinerU。</div>'
                            )
                            auto_index = gr.Checkbox(
                                value=True,
                                label="上传后立即建立向量索引",
                            )
                            auto_graph = gr.Checkbox(
                                value=False,
                                label="上传后立即抽取知识图谱（会调用 LLM）",
                            )
                            use_mineru = gr.Checkbox(
                                value=False,
                                label="使用 MinerU 增强解析（可选，逐文件处理）",
                            )
                            upload_button = gr.Button("导入并处理", variant="primary")
                            operation_output = gr.Markdown("等待导入文档。")
                        with gr.Column(
                            scale=8,
                            elem_classes=["surface-card", "right-workspace"],
                        ):
                            gr.HTML(
                                '<div class="card-heading"><div><span class="icon">库</span>'
                                "<strong>知识资产</strong><p>统一查看解析、向量索引与图谱状态</p>"
                                "</div></div>"
                            )
                            document_table = gr.Dataframe(
                                headers=DOCUMENT_HEADERS,
                                value=[],
                                interactive=False,
                                wrap=True,
                                max_height=440,
                                show_search="search",
                                elem_classes="library-table",
                            )
                            index_selector = gr.Dropdown(
                                choices=[],
                                multiselect=True,
                                label="选择需要处理的文档",
                                info="可同时选择多份文档。",
                            )
                            with gr.Row():
                                index_button = gr.Button("重建向量索引", scale=1)
                                graph_button = gr.Button("抽取知识图谱", scale=1)
                                rebuild_button = gr.Button("后台完整重建", scale=1)
                                delete_document_button = gr.Button(
                                    "删除文档",
                                    scale=1,
                                    elem_classes="danger-button",
                                )
                            confirm_document_delete = gr.Checkbox(
                                value=False,
                                label="确认删除所选文档及其向量和图谱数据",
                            )

                    with gr.Column(elem_classes=["surface-card", "graph-card"]):
                        gr.HTML(
                            '<div class="card-heading"><div><span class="icon">图</span>'
                            "<strong>知识图谱</strong><p>按当前空间、所选文档或实体关键词查看关系网络</p>"
                            "</div></div>"
                        )
                        with gr.Row(elem_classes="manager-row"):
                            graph_query = gr.Textbox(
                                label="实体或关系关键词",
                                placeholder="留空显示当前范围的关系",
                                scale=6,
                            )
                            graph_refresh_button = gr.Button("刷新图谱", scale=1)
                        graph_visualization = gr.HTML(render_graph_visualization({}))

                    with gr.Column(elem_classes=["surface-card", "jobs-card"]):
                        with gr.Row(elem_classes="section-bar"):
                            with gr.Column(scale=8):
                                gr.HTML(
                                    '<div class="card-heading"><div><span class="icon">序</span>'
                                    "<strong>后台任务</strong><p>长耗时重建任务可离开页面继续运行</p>"
                                    "</div></div>"
                                )
                            refresh_jobs_button = gr.Button("刷新任务", size="sm", scale=1)
                        jobs_panel = gr.HTML(render_jobs([]))

                with gr.Column(visible=False) as qa_workspace:
                    with gr.Row(elem_classes="qa-toolbar-grid"):
                        with gr.Column(
                            scale=7,
                            elem_classes=["surface-card", "qa-toolbar-card"],
                        ):
                            gr.HTML(
                                '<div class="card-heading"><div><span class="icon">话</span>'
                                "<strong>会话工作区</strong><p>切换历史会话，记忆会随会话自动恢复</p>"
                                "</div></div>"
                            )
                            qa_knowledge_base_selector = gr.Dropdown(
                                choices=[],
                                label="问答知识空间",
                                info="问答默认只使用该空间内的资料。",
                            )
                            with gr.Row(elem_classes="conversation-actions"):
                                conversation_selector = gr.Dropdown(
                                    choices=[],
                                    label="当前会话",
                                    scale=5,
                                )
                                new_conversation_button = gr.Button(
                                    "＋ 新建会话", variant="primary", size="sm", scale=1
                                )
                                delete_conversation_button = gr.Button(
                                    "删除",
                                    variant="secondary",
                                    size="sm",
                                    scale=1,
                                    elem_classes="danger-button",
                                )
                        with gr.Column(
                            scale=5,
                            elem_classes=["surface-card", "memory-card"],
                        ):
                            memory_panel = gr.HTML(render_memory_status(None))

                    with gr.Row(elem_classes="qa-main-grid"):
                        with (
                            gr.Column(
                                scale=7,
                                elem_classes="qa-main-column",
                            ),
                            gr.Column(elem_classes=["surface-card", "chat-card"]),
                        ):
                            gr.HTML(
                                '<div class="card-heading"><div><span class="icon">答</span>'
                                "<strong>知枢问答</strong><p>回答内容保留文档引用与图谱关系来源</p>"
                                "</div></div>"
                            )
                            chatbot = gr.Chatbot(
                                label="知枢问答",
                                show_label=False,
                                height=720,
                                resizable=True,
                                layout="bubble",
                                placeholder="从你的知识空间开始提问，知枢会检索、判断并组织答案。",
                                buttons=["copy", "copy_all"],
                                watermark="知枢 ZhiShu",
                                elem_id="zhishu-chatbot",
                            )
                        with gr.Column(scale=5, elem_classes="qa-main-column"):
                            with gr.Column(elem_classes=["surface-card", "composer-card"]):
                                gr.HTML(
                                    '<div class="card-heading"><div><span class="icon">问</span>'
                                    "<strong>发起问题</strong><p>设定检索范围与回答方式</p>"
                                    "</div></div>"
                                )
                                with gr.Row(elem_classes="composer-actions"):
                                    question = gr.Textbox(
                                        placeholder="输入问题后按 Enter，或点击发送…",
                                        label="问题",
                                        lines=2,
                                        max_lines=5,
                                        scale=6,
                                    )
                                    ask_button = gr.Button(
                                        "发送 ↗",
                                        variant="primary",
                                        scale=1,
                                        elem_id="zhishu-send",
                                    )
                                with gr.Row():
                                    scope_selector = gr.Dropdown(
                                        choices=[],
                                        multiselect=True,
                                        label="资料范围",
                                        info="留空表示检索当前知识空间的全部资料",
                                        scale=4,
                                    )
                                    top_k = gr.Slider(
                                        minimum=1,
                                        maximum=15,
                                        value=5,
                                        step=1,
                                        label="检索条数",
                                        scale=2,
                                    )
                                rag_mode = gr.Radio(
                                    choices=[AGENTIC_MODE, "基础 RAG"],
                                    value=AGENTIC_MODE,
                                    label="回答模式",
                                    info="智能模式使用会话记忆、向量证据、图谱关系和纠正检索。",
                                )
                                gr.Examples(
                                    examples=[
                                        ["LangGraph 自定义 Agentic RAG 包含哪些主要步骤？"],
                                        ["LangChain Agent 如何使用工具？"],
                                        ["如何为 LangChain 开启 LangSmith tracing？"],
                                    ],
                                    inputs=[question],
                                    label="试试这样问",
                                )
                                clear_button = gr.ClearButton(
                                    [question, chatbot],
                                    value="清空当前显示（不删除会话记忆）",
                                    size="sm",
                                )
                            with gr.Column(elem_classes=["surface-card", "trace-card"]):
                                gr.HTML(
                                    '<div class="card-heading"><div><span class="icon">迹</span>'
                                    "<strong>推理与证据</strong><p>路由、检索轨迹、引用片段和图谱关系</p>"
                                    "</div></div>"
                                )
                                evidence_panel = gr.Markdown(
                                    "### 等待问题\n发送问题后，这里会展示知枢的执行过程与证据。",
                                    elem_classes="evidence-panel",
                                )

        workspace_nav.change(
            lambda selected: (
                gr.update(visible=selected == "知识空间"),
                gr.update(visible=selected == "智能问答"),
            ),
            inputs=[workspace_nav],
            outputs=[knowledge_workspace, qa_workspace],
            show_progress="hidden",
        )

        def refresh_event() -> tuple[str, Any, Any, list[list[Any]], Any, Any, str, str]:
            return refresh_workspace(client)

        refresh_outputs = [
            status_panel,
            knowledge_base_selector,
            qa_knowledge_base_selector,
            document_table,
            index_selector,
            scope_selector,
            graph_visualization,
            jobs_panel,
        ]
        demo.load(refresh_event, outputs=refresh_outputs)
        demo.load(
            lambda: initialize_conversations(client),
            outputs=[conversation_selector, chatbot, memory_panel],
        )
        refresh_button.click(refresh_event, outputs=refresh_outputs)
        build_timer = gr.Timer(5)
        build_timer.tick(
            lambda selected, document_ids, query: poll_build_status(
                selected,
                client,
                document_ids,
                query,
            ),
            inputs=[knowledge_base_selector, index_selector, graph_query],
            outputs=[document_table, jobs_panel, graph_visualization],
            show_progress="hidden",
            concurrency_limit=1,
        )
        knowledge_base_selector.change(
            lambda selected: refresh_assets(selected, client),
            inputs=[knowledge_base_selector],
            outputs=[document_table, index_selector, graph_visualization, jobs_panel],
            show_progress="hidden",
        )
        qa_knowledge_base_selector.change(
            lambda selected: refresh_question_scope(selected, client),
            inputs=[qa_knowledge_base_selector],
            outputs=[scope_selector],
            show_progress="hidden",
        )
        create_base_button.click(
            lambda name: create_knowledge_base(name, client),
            inputs=[knowledge_base_name],
            outputs=[operation_output, *refresh_outputs],
        )
        rename_base_button.click(
            lambda selected, name: rename_knowledge_base(selected, name, client),
            inputs=[knowledge_base_selector, knowledge_base_name],
            outputs=[operation_output, *refresh_outputs],
        )
        delete_base_button.click(
            lambda selected, confirmed: delete_knowledge_base(selected, confirmed, client),
            inputs=[knowledge_base_selector, confirm_base_delete],
            outputs=[operation_output, *refresh_outputs],
        )
        upload_button.click(
            lambda selected_files, should_index, should_graph, should_use_mineru, base_id: (
                upload_documents(
                    selected_files,
                    should_index,
                    should_graph,
                    should_use_mineru,
                    base_id,
                    client,
                )
            ),
            inputs=[
                files,
                auto_index,
                auto_graph,
                use_mineru,
                knowledge_base_selector,
            ],
            outputs=[
                operation_output,
                status_panel,
                document_table,
                index_selector,
                scope_selector,
                graph_visualization,
                jobs_panel,
            ],
        )
        index_button.click(
            lambda selected, base_id: index_selected(selected, base_id, client),
            inputs=[index_selector, knowledge_base_selector],
            outputs=[operation_output, status_panel, document_table, graph_visualization],
        )
        graph_button.click(
            lambda selected, base_id: extract_selected_graph(selected, base_id, client),
            inputs=[index_selector, knowledge_base_selector],
            outputs=[operation_output, status_panel, document_table, graph_visualization],
        )
        rebuild_button.click(
            lambda selected: queue_selected_rebuilds(selected, client),
            inputs=[index_selector],
            outputs=[operation_output, jobs_panel],
        )
        delete_document_button.click(
            lambda selected, base_id, confirmed: delete_selected_documents(
                selected,
                base_id,
                confirmed,
                client,
            ),
            inputs=[index_selector, knowledge_base_selector, confirm_document_delete],
            outputs=[
                operation_output,
                status_panel,
                document_table,
                index_selector,
                scope_selector,
                graph_visualization,
                jobs_panel,
            ],
        )
        graph_refresh_button.click(
            lambda selected, base_id, query: refresh_graph_view(
                selected,
                base_id,
                query,
                client,
            ),
            inputs=[index_selector, knowledge_base_selector, graph_query],
            outputs=[graph_visualization],
        )
        refresh_jobs_button.click(
            lambda: render_jobs(client.list_jobs()),
            outputs=[jobs_panel],
        )
        new_conversation_button.click(
            lambda: create_new_conversation(client),
            outputs=[conversation_selector, chatbot, memory_panel, evidence_panel],
        )
        conversation_selector.change(
            lambda selected: load_conversation(selected, client),
            inputs=[conversation_selector],
            outputs=[chatbot, memory_panel, evidence_panel],
        )
        delete_conversation_button.click(
            lambda selected: delete_current_conversation(selected, client),
            inputs=[conversation_selector],
            outputs=[conversation_selector, chatbot, memory_panel, evidence_panel],
        )

        def ask_event(
            text: str,
            chat_history: list[dict[str, Any]],
            selected_documents: list[str],
            retrieval_top_k: int,
            selected_mode: str,
            selected_session: str,
            selected_knowledge_base: str,
        ) -> tuple[list[dict[str, Any]], str, str, Any, str]:
            return ask_question(
                text,
                chat_history,
                selected_documents,
                retrieval_top_k,
                selected_mode,
                selected_session,
                selected_knowledge_base,
                client,
            )

        ask_inputs = [
            question,
            chatbot,
            scope_selector,
            top_k,
            rag_mode,
            conversation_selector,
            qa_knowledge_base_selector,
        ]
        ask_outputs = [
            chatbot,
            question,
            evidence_panel,
            conversation_selector,
            memory_panel,
        ]
        ask_button.click(ask_event, inputs=ask_inputs, outputs=ask_outputs)
        question.submit(ask_event, inputs=ask_inputs, outputs=ask_outputs)
        clear_button.click(
            lambda: "### 执行详情\n提问后在此查看轨迹、来源和图谱关系。",
            outputs=evidence_panel,
        )

    return demo


def main() -> None:
    settings = get_settings()
    build_ui().queue(default_concurrency_limit=4).launch(
        server_name=settings.app_host,
        server_port=settings.ui_port,
        show_error=True,
        css=CSS,
        theme=THEME,
    )
