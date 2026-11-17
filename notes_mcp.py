"""
MCP server：把「查我的学习笔记」这个能力暴露给任意支持 MCP 的 AI 客户端。

跑起来之后，Claude Desktop / Cursor 这类客户端就能直接调用这里的三个工具，
不需要经过我的 FastAPI 服务，也不需要打开网页。
"""


import logging
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)


import jieba


jieba.setLogLevel(logging.ERROR)

from mcp.server.mcpserver import MCPServer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from search_chroma import search_notes as _search


NOTES_DIR = os.environ.get("NOTES_DIR", r"D:\code\学习笔记")
mcp = MCPServer("notes")


@mcp.tool()
def list_notes()->str:
    names = sorted(f for f in os.listdir(NOTES_DIR) if f.endswith(".md"))
    return "\n".join(names)


@mcp.tool()
def search_notes(query:str, top_k: int = 5)->str:
    hits = _search(query, top_k=top_k)
    if not hits:
        return "没有找到相关内容。"
    parts = []
    for h in hits:
        parts.append(f"【{h['source']}】\n{h['text']}")
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def read_note(filename:str)->str:
    name = os.path.basename(filename.strip())
    real = os.path.realpath(os.path.join(NOTES_DIR, name))
    if not real.startswith(os.path.realpath(NOTES_DIR)):
        return "只能读取笔记目录里的文件。"
    if not os.path.exists(real):
        return f"没有这个文件：{name}"
    with open(real,encoding="utf-8")as f:
        return f.read()

MCP_TOKEN = os.environ.get("MCP_TOKEN","").strip()

class BearerAuth(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path.startswith("/mcp"):
            if request.headers.get("authorization","") != f"Bearer {MCP_TOKEN}":
                return JSONResponse({"error":"unauthorized"}, status_code=401)
        return await call_next(request)


if __name__ == "__main__":
    if "--http" in sys.argv:
        if not MCP_TOKEN:
            raise SystemExit("远程模式必须先设置MCP_TOJEN环境变量，否则谁都能查你的笔记")
        import uvicorn
        app = mcp.streamable_http_app()
        app.add_middleware(BearerAuth)
        uvicorn.run(app, host = "0.0.0.0",
                    port = int(os.environ.get("MCP_PROT","9000")),
                    log_level = "warning")
    else:
        mcp.run(transport = "stdio")