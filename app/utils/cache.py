from __future__ import annotations
from collections import OrderedDict
import json, sqlite3, time, os, threading
from typing import Any, Optional, Tuple

class LRUCache:
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.store = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            if key in self.store:
                self.store.move_to_end(key)
                return self.store[key]
            return None

    def set(self, key: str, value: Any):
        with self.lock:
            self.store[key] = value
            self.store.move_to_end(key)
            if len(self.store) > self.capacity:
                self.store.popitem(last=False)

class SQLiteCache:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._init()
        self.lock = threading.Lock()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS cache (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL,
            created_at REAL NOT NULL
        )""")
        self.conn.commit()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT v FROM cache WHERE k=?", (key,))
            row = cur.fetchone()
            if not row: return None
            try:
                return json.loads(row[0])
            except Exception:
                return None

    def set(self, key: str, value: Any):
        s = json.dumps(value, separators=(',',':'))
        ts = time.time()
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("REPLACE INTO cache (k,v,created_at) VALUES (?,?,?)", (key, s, ts))
            self.conn.commit()
