"""Действия агента.

Реальных внутренних API (биллинг, сессии) у учебного проекта нет, поэтому агент
не делает вид, что выполнил операцию: он создаёт заявку в таблице tickets и
сообщает клиенту её номер. Это честная точка интеграции — заменить запись
в таблицу на вызов настоящего API можно, не трогая граф.
"""

from src.db import get_conn


def create_ticket(category: str, user_id: str, text: str) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO tickets (category, user_id, text) VALUES (%s, %s, %s) RETURNING id;",
            (category, user_id, text),
        )
        return int(cur.fetchone()[0])
