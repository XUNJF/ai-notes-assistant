# test.py
import multi_agent as M

cfg = {"configurable": {"thread_id": "regress-1"}}
for q in ["我是小明", "RAG 是什么？", "8472+3951 是多少", "我是谁"]:
    out = M.graph.invoke({"question": q}, config=cfg)
    print(f"  {q}")
    print(f"     资料={len(out.get('notes') or '')}字  计算={out.get('calc_result') or '无'}")
    print(f"     答={(out.get('answer') or '')[:50]}")