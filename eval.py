"""
评测：跑完整链路，量三件事
  ① 路由正确率 —— 主管有没有分派对（闲聊 / 计算 / 检索）
  ② 检索命中率 —— 该查笔记时，期望的那份笔记有没有出现在 sources 里
  ③ 端到端耗时
"""

import time 

import multi_agent as M

CASES = [
    ("你好",                       "chat",     None),
    ("今天脑子有点乱",               "chat",     None),
    ("谢谢",                       "chat",     None),

    ("8472+3951 是多少",            "calc",     None),
    ("25 乘以 16 再减 137 等于多少",  "calc",     None),
    ("2 的 10 次方是多少",           "calc",     None),

    ("变量是什么",                   "research", "第3天-学习笔记.md"),
    ("input 怎么用",                "research", "第3天-学习笔记.md"),
    ("列表索引从几开始",             "research", "第4天-学习笔记.md"),
    ("字典怎么取值",                 "research", "第5天-学习笔记.md"),
    ("return 和 print 有什么区别",   "research", "第6天-学习笔记.md"),
    ("读文件为什么要写 encoding",    "research", "第7天-学习笔记.md"),
    ("虚拟环境是干什么的",           "research", "第8天-学习笔记.md"),
    ("FastAPI 怎么定义一个接口",     "research", "第13天-学习笔记.md"),
    ("Embedding 是什么",            "research", "第17天-学习笔记.md"),
    ("RAG 是什么",                  "research", "第18天-学习笔记.md"),
]


def actual_route(out):
    if out.get("calc_result"):
        return "calc"
    if out.get("notes"):
        return "research"
    return "chat"

def run(question,thread_id):
    cfg = {"configurable":{"thread_id":thread_id}}
    t0 = time.time()
    out = M.graph.invoke({"question":question},config=cfg)
    return out, time.time() - t0


def main():
    ok_route = 0
    ok_hit = 0
    need_hit = 0
    total_time = 0.0

    print(f"  共 {len(CASES)} 题\n")
    print(f"  {'问题':<26} {'期望':<9}{'实际':<10}{'命中':<5}{'耗时':>6}")
    print(f"  {'-' * 60}")

    for i, (q,expect, expect_src) in enumerate(CASES):
        out, cost = run(q,f"eval-{i}")
        total_time += cost
        actual = actual_route(out)
        if actual == expect:
            ok_route += 1
        if expect == "research":
            need_hit += 1
            hit = expect_src in (out.get("sources") or [])
            if hit:
                ok_hit += 1
            mark = "✓" if hit else "✗"
        else:
            mark = "-"
        print(f"  {q:<26} {expect:<9}{actual:<10}{mark:<5}{cost:>5.1f}s")
    n = len(CASES)
    print(f"\n  路由正确率：{ok_route}/{n} = {ok_route / n:.0%}")
    print(f"  检索命中率：{ok_hit}/{need_hit} = {ok_hit / need_hit:.0%}")
    print(f"  平均耗时：  {total_time / n:.2f}s")
    cfg = {"configurable":{"thread_id":"eval-memory"}}
    M.graph.invoke({"question":"我叫小明"},config=cfg)
    out = M.graph.invoke({"question": "我叫什么名字"}, config=cfg)
    ans = out.get("answer") or ""
    print(f"  记忆测试：  {'✓ 记住了' if '小明' in ans else '✗ 忘了'}（答：{ans[:30]}）")

if __name__ == "__main__":
    main()

