from typing import TypedDict,Annotated

from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph,START,END,add_messages

from rag_agent import _read_key
from search_chroma import search_notes,build_context
import ast 
import operator
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import RemoveMessage



_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def safe_eval(expr):
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value,(int,float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left),walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"不支持的写法：{type(node).__name__}")
    return walk(ast.parse(expr, mode="eval").body)
        

MAX_MESSAGES = 4
KEEP_RECENT = 2


ds_key = _read_key("DEEPSEEK_API_KEY","key.txt")
model = ChatDeepSeek(model="deepseek-flash", api_key=ds_key, timeout=60, max_retries=3,temperature=0)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    notes: str
    answer: str
    next: str
    sources: list
    calc_result:str
    summary: str

SUPERVISOR_PROMPT = """你是主管，手下有三个专员：
- research   研究员：翻学习笔记，找原始资料
- calc       计算员：做算术题（加减乘除、乘方）
- write      撰稿人：根据已有信息写答案

看当前进度，决定下一步派谁。只回一个词。

什么时候派 research：问题跟他的学习内容有关，而且还没查过资料
什么时候派 calc：    问题是纯算术（算个数、四则运算、乘方之类）
什么时候派 write：   资料/计算结果已经有了，或者这问题什么都不用查（闲聊、问"我是谁"）"""

def receive(state):
    return {
        "messages":[{"role":"user","content":state["question"]}],
        "notes":"",
        "answer":"",
        "sources":[],
        "next":"",
        "calc_result":"",
    }

def compress(state):
    msgs = state.get("messages",[])
    if len(msgs) <= MAX_MESSAGES:
        return {}
    old = msgs[:-KEEP_RECENT]
    raw = "\n".join(
        f"{'用户' if m.type == 'human' else '助手'}:{m.content}"for m in old
    )
    before = state.get("summary","")
    prompt = (f"【已有摘要】\n{before}\n\n【新的对话】\n{raw}\n\n"
              if before else f"【对话】\n{raw}\n\n")
    r = model.invoke([
        {"role":"system","content":
        "把下面这段对话压缩成一段简短摘要。\n"
         "【必须保留】\n"
         "1. 用户的个人信息：姓名、身份、偏好、习惯、约定过的事（一个字都不能丢）\n"
         "2. 用户问了什么、结论是什么\n"
         "3. 关键概念\n"
         "不要评论，不要分段，200 字以内。"},
        {"role":"user","content":prompt},
    ])
    return {
        "summary":r.content.strip(),
        "messages":[RemoveMessage(id = m.id) for m in old],
    }



def supervisor(state):
    status = (f"当前进度：资料={'有' if state.get('notes') else '没有'},"
              f"计算结果={'有' if state.get('calc_result') else '没有'},"
              f"答案={'有' if state.get('answer') else '没有'}")
    msgs = [{"role":"system","content":SUPERVISOR_PROMPT}]
    if state.get("summary"):
        msgs.append({"role":"system",
                     "content":f"【之前聊过的内容摘要】\n{state['summary']}"})
    msgs.extend(state.get("messages",[])[-6:])
    msgs.append({"role":"user","content":status})
    r = model.invoke(msgs)
    return {"next":r.content.strip().lower()}
    

def researcher(state):
    hits = search_notes(state['question'])
    text, sources = build_context(hits)
    return {"notes":text, "sources":sources}

def calculator(state):
    expr = model.invoke([
        {"role":"system","content":"把用户的算术题转成一个纯算式，只输出算式，不要等号、不要解释。用 + - * / ** 表示。"},
        {"role":"user","content":state['question']},
    ]).content.strip().strip("`")
    try:
        result =safe_eval(expr)
        return {"calc_result":f"{expr} = {result}"}
    except Exception as e:
        return {"calc_result":f"算不出来（{e}）"}



def writer(state):
    notes = state.get("notes") or ""
    calc = state.get("calc_result") or ""
    if calc:
        system = f"把计算结果告诉用户，简短说明即可。\n【计算结果】{calc}"
    elif notes:
        system = ("只根据下面提供的【资料】回答；资料里没有就说「资料里没有提到这个」。\n\n"
                f"【资料】\n{notes}")
    else:
        system = ("根据前面的对话自然回应。\n"
                  "如果这是个技术问题，而你没有拿到相关资料，"
                  "就明确说「笔记里没有相关内容」——不要凭自己的知识编。\n"
                  "闲聊就简短回应。")
    msgs = [{"role":"system","content":system}]
    if state.get("summary"):
        msgs.append({"role":"system","content": f"【之前聊过的内容摘要】\n{state['summary']}"})
    msgs.extend(state.get("messages",[]))
    r = model.invoke(msgs)
    return {
        "answer":r.content,
        "messages":[{"role":"assistant","content":r.content}],
    }


def route(state):
    n = state.get("next","")
    if "research" in n:
        return "researcher"
    if "calc" in n:
        return "calculator"
    return "writer"

builder = StateGraph(State)
builder.add_node("compress",compress)
builder.add_node("receive",receive)  
builder.add_node("supervisor",supervisor)
builder.add_node("researcher",researcher)
builder.add_node("calculator",calculator)
builder.add_node("writer",writer)

builder.add_edge(START,"compress")
builder.add_edge("compress","receive")            # ← START 先到 receive
builder.add_edge("receive","supervisor") 
builder.add_conditional_edges(
    "supervisor",
    route,
    {
        "researcher":"researcher",
        "calculator":"calculator",
        "writer":"writer",
    }
)

builder.add_edge("researcher","supervisor")
builder.add_edge("calculator","supervisor")
builder.add_edge("writer",END)

os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/记忆.db", check_same_thread=False)
graph = builder.compile(checkpointer=SqliteSaver(conn))


if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "debug1"}}

    for q in ["我是小明", "我是谁？", "8472加3951再乘27减1364等于多少", "RAG 是什么？"]:
        print("=" * 60)
        print(f"  问题：{q}")
        print("=" * 60)
        out = graph.invoke({"question": q}, config=cfg)
        print(f"  主管决定：{out.get('next')}")
        print(f"  查过资料：{'是' if out.get('notes') else '否'}")
        print(f"  计算结果：{out.get('calc_result') or '无'}")
        print(f"  答案：{out.get('answer', '')[:100]}")
        print()
