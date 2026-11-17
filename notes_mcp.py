"""
MCP server：把「查我的学习笔记」这个能力暴露给任意支持 MCP 的 AI 客户端。

跑起来之后，Claude Desktop / Cursor 这类客户端就能直接调用这里的三个工具，
不需要经过我的 FastAPI 服务，也不需要打开网页。
"""


import logging
import os

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)


import jieba


jieba.setLogLevel(logging.ERROR)

from mcp.server.mcpserver import MCPServer
from search_chroma import search_notes as _search

NOTES_DIR =r"D:\code\学习笔记"
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

if __name__ == "__main__":
    mcp.run(transport="stdio")