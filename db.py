import json
import sqlite3

DB = "history.db"

def connect():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def create_table():
    c = connect()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history(
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id   TEXT NOT NULL,
            question    TEXT NOT NULL,
            answer      TEXT NOT NULL,
            sources     TEXT,
            created     TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.commit()
    c.close()

def save(question, answer, sources, thread_id="default"):
    c = connect()
    c.execute(
        "INSERT INTO history (thread_id, question, answer, sources) VALUES (?,?,?,?)",
        (thread_id, question, answer, json.dumps(sources, ensure_ascii=False)),
    )
    c.commit()
    c.close()

def recent(n=20):
    c = connect()
    cur = c.execute(
        "SELECT id, thread_id, question, answer, sources, created FROM history ORDER BY id DESC LIMIT ?",
        (n,),
    )
    rows = []
    for r in cur.fetchall():
        d = dict(r)
        d["sources"] = json.loads(d["sources"]) if d["sources"] else []
        rows.append(d)
    c.close()
    return rows

def sessions(n=20):
    c = connect()
    cur = c.execute("""
        SELECT thread_id,
        COUNT(*) AS n,
        MAX(created) AS last_time,
        (SELECT h2.question FROM history h2
            WHERE h2.thread_id = h1.thread_id
            ORDER BY h2.id ASC LIMIT 1) AS title
        FROM history h1
        GROUP BY thread_id
        ORDER BY MAX(id) DESC
        LIMIT ?
    """,(n,))
    rows = [dict(r) for r in cur.fetchall()]
    c.close()
    return rows
    
def by_thread(thread_id):
    c = connect()
    cur = c.execute(
        "SELECT id, question, answer, sources, created FROM history WHERE thread_id = ? ORDER BY id ASC",
        (thread_id,),
    )
    rows = []
    for r in cur.fetchall():
        d = dict(r)
        d["sources"] = json.loads(d["sources"]) if d["sources"] else []
        rows.append(d)
    c.close()
    return rows


def count():
    c = connect()
    n = c.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    c.close()
    return n


if __name__ == "__main__":
    create_table()
    print("建表完成，现在有", count(), "条")
    save("测试问题", "测试回答", ["第17天-学习笔记.md"], "demo")
    print("存 1 条，现在有", count(), "条")
    for r in recent(3):
        print(f"  [{r['id']}] {r['thread_id']} | {r['question']} → {r['answer']}")
        print(f"       来源：{r['sources']}")