"""Local SQLite signature library.

Solved signals can be saved (CLI --save-signature, GUI) and listed. New
analyses are not matched against it."""
from __future__ import annotations

import json
import os
import sqlite3
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signatures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created REAL NOT NULL,
    modulation TEXT,
    symbol_rate_norm REAL,
    bandwidth_norm REAL,
    fec_family TEXT,
    fec_params TEXT,
    interleaver_kind TEXT,
    interleaver_params TEXT,
    scrambler TEXT,
    frame_length_bits INTEGER,
    sync_word_hex TEXT,
    crc_name TEXT,
    notes TEXT
);
"""


class SignatureDB:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def save_from_result(self, result, name: str, notes: str = "") -> int:
        r = result.to_dict() if hasattr(result, "to_dict") else result
        params = r.get("parameters") or {}
        fec = r.get("fec") or {}
        il = r.get("interleaver") or {}
        fr = r.get("frames") or {}
        cur = self.conn.execute(
            """INSERT INTO signatures (name, created, modulation,
               symbol_rate_norm, bandwidth_norm, fec_family, fec_params,
               interleaver_kind, interleaver_params, scrambler,
               frame_length_bits, sync_word_hex, crc_name, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (name, time.time(),
             (r.get("modulation") or {}).get("prediction"),
             params.get("symbol_rate_norm_recording") or
             params.get("symbol_rate_norm"), params.get("obw99_norm"),
             fec.get("family"), json.dumps(fec.get("parameters", {})),
             il.get("kind"), json.dumps(il.get("parameters", {})),
             (r.get("scrambler") or {}).get("name"),
             fr.get("frame_length_bits"), fr.get("sync_word_hex"),
             (fr.get("crc") or {}).get("name") if fr.get("crc") else None,
             notes))
        self.conn.commit()
        return cur.lastrowid

    def list_all(self) -> list:
        rows = self.conn.execute(
            "SELECT id, name, modulation, sync_word_hex, created"
            " FROM signatures ORDER BY created DESC").fetchall()
        return [{"id": r[0], "name": r[1], "modulation": r[2],
                 "sync_word_hex": r[3], "created": r[4]} for r in rows]
