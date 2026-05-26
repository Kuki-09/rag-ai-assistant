import os
import sqlite3
from threading import local

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "memory.db")


class ChatMemory:
    """
    Thread-safe SQLite-backed chat memory.
    Uses threading.local() so each thread gets its own connection.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._local = local()
        # Ensure table exists (run once on the main thread)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id    TEXT    NOT NULL,
                query     TEXT    NOT NULL,
                response  TEXT    NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._local.conn

    def save(self, doc_id: str, query: str, response: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO chat (doc_id, query, response) VALUES (?, ?, ?)",
            (doc_id, query, response),
        )
        conn.commit()

    def get_history(self, doc_id: str, limit: int = 5) -> list[tuple]:
        """
        Return the last `limit` turns in CHRONOLOGICAL order (oldest first).
        ✅ FIX: was DESC, giving the LLM reversed context.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT query, response FROM chat
            WHERE doc_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (doc_id, limit),
        )
        rows = cursor.fetchall()
        rows.reverse()  # oldest first for LLM context
        return rows

    def clear(self, doc_id: str) -> None:
        """Clear history for a specific doc."""
        conn = self._get_conn()
        conn.execute("DELETE FROM chat WHERE doc_id = ?", (doc_id,))
        conn.commit()