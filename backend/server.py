"""合并版后端 SSE 服务：一个文件覆盖 P1–P6 六个阶段的全部实战。

运行模式由环境变量 MODE 决定（默认 echo，即 P1 阶段）：

    echo    P1  · SSE 回声：不需要任何 Key，先通水管
    p3      P3  · 离线两轮 ReAct mock（mock_retrieval.plan），无需 Key
    p4      P4  · 离线 MCP 检索 + A2A 委派 mock，无需 Key
    agent        真 AgentScope 2.0.8 Agent（需要模型 Key；P4 工具需同时起 mcp_server / a2a_server）

SSE 事件（同一套，前端一份渲染代码通吃四个模式）：
    status       {"stage": "start"}
    think        {"content": "…"}          思维链增量
    tool_call    {"tool": "…", "args": {}}     # MCP / 本地工具
    tool_result  {"tool": "…", "text": "…"}
    agent_call   {"agent": "…", "args": {}}   # A2A 子 Agent
    agent_result {"agent": "…", "text": "…"}
    delta        {"content": "字"}          答案增量
    done         {"content": "整段答案"}
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).with_name(".env"))

MODE = os.environ.get("MODE", "echo").strip().lower()
if MODE not in ("echo", "p3", "p4", "agent"):
    raise SystemExit(f"MODE 只能是 echo / p3 / p4 / agent，当前是 {MODE!r}")

MODE_LABEL = {"echo": "echo", "p3": "p3-mock", "p4": "p4-mock", "agent": "agent"}

app = FastAPI(title="高代码实战 · 合并版 Agent SSE", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# session_id → {question, conversation_id}；流结束后丢掉（P1 教学用内存会话）
SESSIONS: dict[str, dict[str, str]] = {}
# conversation_id → Agent（真模式才用，跨多轮记忆）
AGENTS: dict[str, Any] = {}


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class ChatOut(BaseModel):
    session_id: str
    conversation_id: str


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _etype(ev: Any) -> str:
    t = getattr(ev, "type", "")
    return t.value if hasattr(t, "value") else str(t)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "mode": MODE_LABEL[MODE], "conversations": len(AGENTS)}


@app.post("/api/chat", response_model=ChatOut)
async def chat(body: ChatIn) -> ChatOut:
    sid = uuid.uuid4().hex
    cid = (body.conversation_id or "").strip() or uuid.uuid4().hex
    SESSIONS[sid] = {"question": body.message.strip(), "conversation_id": cid}
    return ChatOut(session_id=sid, conversation_id=cid)


@app.post("/api/conversations/{conversation_id}/reset")
def reset_conversation(conversation_id: str) -> dict[str, Any]:
    AGENTS.pop(conversation_id, None)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# P1 · echo：只通水管，不碰模型
# --------------------------------------------------------------------------- #
async def echo_stream(text: str) -> AsyncGenerator[str, None]:
    payload = f"回声：{text}"
    yield sse("delta", {"content": ""})
    for ch in payload:
        yield sse("delta", {"content": ch})
        await asyncio.sleep(0.04)
    yield sse("done", {"content": payload})


# --------------------------------------------------------------------------- #
# P3 · 离线两轮 ReAct（mock_retrieval.plan 产出工具调用序列）
# --------------------------------------------------------------------------- #
async def p3_stream(question: str) -> AsyncGenerator[str, None]:
    from mock_retrieval import plan

    yield ": ping\n\n"
    yield sse("status", {"stage": "start"})
    yield sse("think", {"content": "先判断要不要查合同，再决定是否计算。\n"})
    await asyncio.sleep(0.15)

    for ev in plan(question):
        if ev["type"] == "tool_call":
            yield sse("tool_call", {"tool": ev["tool"], "args": ev.get("args", {})})
            await asyncio.sleep(0.25)
        elif ev["type"] == "tool_result":
            yield sse("tool_result", {"tool": ev["tool"], "text": ev["text"]})
            await asyncio.sleep(0.2)
        elif ev["type"] == "answer":
            text = ev["text"]
            for ch in text:
                yield sse("delta", {"content": ch})
                await asyncio.sleep(0.012)
            yield sse("done", {"content": text})


# --------------------------------------------------------------------------- #
# P4 · 离线 MCP 检索 + A2A 风险评估两个协议步骤
# --------------------------------------------------------------------------- #
async def p4_stream(question: str) -> AsyncGenerator[str, None]:
    yield ": ping\n\n"
    yield sse("status", {"stage": "start"})
    yield sse("think", {"content": "先通过 MCP 检索合同条款，再决定是否需要 A2A 风险评估。\n"})
    await asyncio.sleep(0.15)

    # 1. MCP 工具调用
    yield sse("tool_call", {"tool": "search_contract", "args": {"question": question}})
    await asyncio.sleep(0.25)
    yield sse(
        "tool_result",
        {
            "tool": "search_contract",
            "text": "[1] 出处：采购合同.pdf / 第 2 页\n第六条：验收合格后 30 日内付款。\n[2] 出处：采购合同.pdf / 第 3 页\n违约金：每逾期一日按合同总价的万分之五计算。",
        },
    )
    await asyncio.sleep(0.2)

    # 2. A2A 子 Agent 委派（仅当问题涉及风险审查）
    if any(k in question for k in ("风险", "审", "合规", "问题")):
        yield sse("agent_call", {"agent": "compliance_agent", "args": {"input": question}})
        await asyncio.sleep(0.25)
        yield sse(
            "agent_result",
            {
                "agent": "compliance_agent",
                "text": "[高] 未明确违约金或赔偿条款，逾期追偿缺乏合同依据。\n[中] 争议解决/管辖条款缺失。",
            },
        )
        await asyncio.sleep(0.2)
        answer = (
            "根据检索结果与合规 Agent 评估：\n"
            "• 付款期限为验收合格后 30 日内（出处：第六条 / 第 2 页）。\n"
            "• [高] 未明确违约金或赔偿条款，逾期追偿缺乏合同依据。\n"
            "• [中] 争议解决/管辖条款缺失，建议补充。"
        )
    else:
        answer = "验收合格后 30 日内付款（出处：采购合同.pdf 第六条 / 第 2 页）。"

    for ch in answer:
        yield sse("delta", {"content": ch})
        await asyncio.sleep(0.012)
    yield sse("done", {"content": answer})


# --------------------------------------------------------------------------- #
# 真 Agent：reply_stream 把思维链 / 工具 / 正文立刻推出去
# --------------------------------------------------------------------------- #
async def agent_stream(question: str, conversation_id: str) -> AsyncGenerator[str, None]:
    yield ": ping\n\n"
    yield sse("status", {"stage": "start"})

    try:
        from agentscope.message import UserMsg

        from agent import build_agent
    except Exception as exc:  # noqa: BLE001
        yield sse("tool_result", {"tool": "system", "text": f"未能加载 AgentScope（{exc}），回退到离线 mock。"})
        async for frame in p3_stream(question):
            if frame.startswith("event: status") or frame.startswith(": ping"):
                continue
            yield frame
        return

    try:
        agent = AGENTS.get(conversation_id)
        if agent is None:
            agent = build_agent()
            AGENTS[conversation_id] = agent
    except Exception as exc:  # noqa: BLE001
        yield sse("tool_result", {"tool": "system", "text": f"创建 Agent 失败（{exc}），回退到离线 mock。"})
        async for frame in p3_stream(question):
            if frame.startswith("event: status") or frame.startswith(": ping"):
                continue
            yield frame
        return

    acc = ""
    pending_args: dict[str, str] = {}
    pending_name: dict[str, str] = {}
    result_buf: dict[str, str] = {}
    result_name: dict[str, str] = {}

    try:
        async for ev in agent.reply_stream(UserMsg(name="user", content=question)):
            kind = _etype(ev)

            if kind == "THINKING_BLOCK_DELTA":
                piece = getattr(ev, "delta", "") or ""
                if piece:
                    yield sse("think", {"content": piece})

            elif kind == "TEXT_BLOCK_DELTA":
                piece = getattr(ev, "delta", "") or ""
                if piece:
                    acc += piece
                    yield sse("delta", {"content": piece})

            elif kind == "TOOL_CALL_START":
                tid = getattr(ev, "tool_call_id", "")
                name = getattr(ev, "tool_call_name", "tool")
                pending_args[tid] = ""
                pending_name[tid] = name

            elif kind == "TOOL_CALL_DELTA":
                tid = getattr(ev, "tool_call_id", "")
                pending_args[tid] = pending_args.get(tid, "") + (getattr(ev, "delta", "") or "")

            elif kind == "TOOL_CALL_END":
                tid = getattr(ev, "tool_call_id", "")
                name = pending_name.get(tid, "tool")
                raw = pending_args.get(tid, "")
                try:
                    args = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    args = {"raw": raw}
                # P4：compliance_review 走 A2A，显示为 agent_call
                if name == "compliance_review":
                    yield sse("agent_call", {"agent": "compliance_agent", "args": args})
                else:
                    yield sse("tool_call", {"tool": name, "args": args})

            elif kind == "TOOL_RESULT_START":
                tid = getattr(ev, "tool_call_id", "")
                result_buf[tid] = ""
                result_name[tid] = getattr(ev, "tool_call_name", "tool")

            elif kind == "TOOL_RESULT_TEXT_DELTA":
                tid = getattr(ev, "tool_call_id", "")
                result_buf[tid] = result_buf.get(tid, "") + (getattr(ev, "delta", "") or "")

            elif kind == "TOOL_RESULT_END":
                tid = getattr(ev, "tool_call_id", "")
                name = result_name.get(tid, pending_name.get(tid, "tool"))
                text = result_buf.get(tid, "")
                # P4：compliance_review 走 A2A，显示为 agent_result
                if name == "compliance_review":
                    yield sse("agent_result", {"agent": "compliance_agent", "text": text})
                else:
                    yield sse("tool_result", {"tool": name, "text": text})

            elif kind == "REPLY_END":
                err = getattr(ev, "error", None)
                if err is not None:
                    msg = getattr(err, "message", None) or str(err)
                    acc = acc or f"Agent 结束出错：{msg}"
                    yield sse("delta", {"content": acc if not acc else ""})
    except Exception as exc:  # noqa: BLE001
        piece = f"Agent 运行出错：{exc}"
        acc = acc or piece
        yield sse("delta", {"content": piece})

    yield sse("done", {"content": acc})


@app.get("/api/stream/{session_id}")
async def stream(session_id: str) -> StreamingResponse:
    rec = SESSIONS.get(session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown session")

    question = rec["question"]
    cid = rec["conversation_id"]

    async def gen() -> AsyncGenerator[str, None]:
        try:
            if MODE == "echo":
                async for frame in echo_stream(question):
                    yield frame
            elif MODE == "p3":
                async for frame in p3_stream(question):
                    yield frame
            elif MODE == "p4":
                async for frame in p4_stream(question):
                    yield frame
            else:
                async for frame in agent_stream(question, cid):
                    yield frame
        finally:
            SESSIONS.pop(session_id, None)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "8000"))
    print(f"启动主服务（MODE={MODE} → {MODE_LABEL[MODE]}）：http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
