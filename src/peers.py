from __future__ import annotations

import sqlite3


def build_peer_groups(conn: sqlite3.Connection) -> None:
    """Заполняет peer_groups: same (post, department).

    Без fallback'а на department-only: руководитель и оператор — концептуально разные классы,
    и сравнивать их в одной peer-группе неправильно. Если same-post группа пуста (единственный
    представитель класса), peer-агрегаты получат 'none'.
    """
    emps = list(
        conn.execute("SELECT employee_id, post, department FROM dim_employee").fetchall()
    )
    for e in emps:
        same_post = [
            r["employee_id"]
            for r in conn.execute(
                """SELECT employee_id FROM dim_employee
                   WHERE post = ? AND department = ? AND employee_id != ?""",
                (e["post"], e["department"], e["employee_id"]),
            ).fetchall()
        ]
        for p in same_post:
            conn.execute(
                "INSERT OR IGNORE INTO peer_groups (employee_id, peer_employee_id) VALUES (?, ?)",
                (e["employee_id"], p),
            )
    conn.commit()


def get_peers(conn: sqlite3.Connection, employee_id: str) -> list:
    return [
        r["peer_employee_id"]
        for r in conn.execute(
            "SELECT peer_employee_id FROM peer_groups WHERE employee_id = ?", (employee_id,)
        ).fetchall()
    ]
