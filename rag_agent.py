"""
笔记问答 Agent（图 + 记忆 + 历史压缩）

给 server.py 用的：它只认 chat(问题, 会话id) 这一个函数。
"""
import os
import json
import requests
import math
from typing import TypedDict, Annotated

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import RemoveMessage
from search_chroma import search_notes,build_context



def _read_key(env_name,filename):
    v = os.environ.get(env_name)
    if v:
        return v.strip()
    try:
        with open(filename,encoding="utf-8")as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(
            f"找不到 {env_name}。\n"
            f"  Docker 里：检查 docker run 有没有带 --env-file .env"
        )


ds_key = _read_key("DEEPSEEK_API_KEY", "key.txt")
sf_key = _read_key("SILICONFLOW_API_KEY", "key_sf.txt")


model = ChatDeepSeek(model="deepseek-flash", api_key=ds_key, timeout=60, max_retries=3,temperature=0)


INDEX_FILE = "index.json"
TOP_K =5


def embed_batch(texts):
    response = requests.post(
        "https://api.siliconflow.cn/v1/embeddings",
        headers={"Authorization":f"Bearer {sf_key}","Content-Type":"application/json"},
        json={"model":"BAAI/bge-m3","input":texts},
        timeout=60,
    )
    d = response.json()
    if d.get("data") is None:
        raise RuntimeError(f"embedding 接口出错:{d}")
    vectors = [None] * len(texts)
    for item in d["data"]:
        vectors[item["index"]] = item["embedding"]
    return vectors


MAX_MESSAGES = 12
KEEP_RECENT = 6


class State(TypedDict):
    messages : Annotated[list, add_messages]
    question : str
    summary : str
    decision : str
    context : str
    sources : list
    answer : str


def compress(state):
    msgs = state["messages"]
    if len(msgs) <= MAX_MESSAGES:
        return {}
    old = msgs[:-KEEP_RECENT]
    原文 = "\n".join(
        f"{'用户' if m.type == 'human' else '助手'}:{m.content}" for m in old
    )
    已有 = state.get("summary","")
    提示 = (
        f"【已有摘要】\n{已有}\n\n【新的对话】\n{原文}\n\n"
        if 已有 else f"【对话】\n{原文}\n\n"
    )
    reply = model.invoke([
        {"role":"system","content":
        "把下面这段对话压缩成一段简短摘要。\n"
        "只保留：用户问了什么、结论是什么、有哪几个关键概念。\n"
        "不要评论，不要分段，200 字以内。"},
        {"role":"user","content":提示},
    ])
    return {
        "summary":reply.content.strip(),
        "messages":[RemoveMessage(id = m.id)for m in old],
    }


def decide(state):
    rules = (
        "判断用户的问题需不需要查他的学习笔记。\n"
        "回答只能用下面两个词之一，不要标点、不要解释：\n"
        "  search —— 当问题是【知识点、概念、技术、怎么用】这类\n"
        "  chat   —— 当问题是【打招呼、闲聊、算数、别的杂事】\n"
    )
    reply = model.invoke([
        {"role":"system","content":rules},
        {"role":"user","content":state['question']},
    ])
    return {
        "decision":reply.content.strip().lower(),
        "messages":[{"role":"user","content":state['question']}],
    }


def route(state):
    return "need_search" if "search" in state["decision"] else "no_search"

def retrieve(state):
    hits = search_notes(state["question"])
    text , sources = build_context(hits)
    return {"context":text,"sources":sources}


def 拼历史(state):
    out = []
    if state.get("summary"):
        out.append({"role":"system","content":f"【之前聊过的内容摘要】\n{state['summary']}"})
    out += list(state["messages"])
    return out

def answer_with_notes(state):
    rules = (
        "你是一个学习助手。只根据提供的【资料】回答。\n"
        "资料里没有就直说「笔记里没有提到」。回答时注明来源。"
    )
    reply = model.invoke(
        [{"role":"system","content":rules}]
        + 拼历史(state)
        + [{"role":"user","content":f"【资料】\n{state['context']}"}]
    )
    return {
        "answer":reply.content,
        "messages":[{"role":"assistant","content":reply.content}]}


def answer_directly(state):
    reply = model.invoke(
        [{"role":"system","content":"你是一个学习助手，回答简短点。"}]
        + 拼历史(state)
    )
    return {
        "answer":reply.content,
        "messages":[{"role":"assistant","content":reply.content}]
    }

builder = StateGraph(State)
builder.add_node("compress",compress)
builder.add_node("decide",decide)
builder.add_node("retrieve",retrieve)
builder.add_node("answer_with_notes",answer_with_notes)
builder.add_node("answer_directly",answer_directly)

builder.add_edge(START,"compress")
builder.add_edge("compress","decide")
builder.add_conditional_edges("decide",route,{"need_search":"retrieve","no_search":"answer_directly"})
builder.add_edge("retrieve","answer_with_notes")
builder.add_edge("answer_with_notes",END)
builder.add_edge("answer_directly",END)

os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/记忆.db", check_same_thread=False)
graph = builder.compile(checkpointer=SqliteSaver(conn))

def chat(question, thread_id = "default"):
    config = {"configurable":{"thread_id":thread_id}}
    out = graph.invoke({"question":question},config=config)
    return {
        "answer":out["answer"],
        "sources":out.get("sources",[]),
        "used_notes":out.get("decision","").startswith("search"),
    }

def chat_stream(question, thread_id="default"):    # 流式版：一块一块吐回答
    """一块一块吐出回答文字，最后给出 (sources, used_notes)"""
    config = {"configurable": {"thread_id": thread_id}}

    for chunk, meta in graph.stream(
        {"question": question}, config=config, stream_mode="messages"
    ):
        # 只转发【回答节点】产生的字，decide 节点也在调模型，那些不要
        node = meta.get("langgraph_node")
        if node in ("answer_with_notes", "answer_directly") and chunk.content:
            yield chunk.content

    state = graph.get_state(config).values
    return state.get("sources", []), state.get("decision", "").startswith("search")
