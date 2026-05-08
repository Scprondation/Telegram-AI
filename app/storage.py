from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class HistoryMessage:
    role: str
    content: str


@dataclass(slots=True)
class DiarySourceMessage:
    chat_label: str
    direction: str
    sender_name: str | None
    text: str
    created_at: str


@dataclass(slots=True)
class DiaryEntry:
    entry_date: str
    summary: str
    source_message_count: int
    updated_at: str


@dataclass(slots=True)
class StoredMessage:
    direction: str
    text: str
    created_at: str


@dataclass(slots=True)
class UserProfile:
    chat_id: int
    chat_label: str
    rating: int
    notes: str
    updated_at: str


@dataclass(slots=True)
class ChatMemory:
    chat_id: int
    chat_label: str
    inferred_gender: str
    known_facts: str
    communication_style: str
    relationship_summary: str
    recent_events: str
    affection: int
    trust: int
    loyalty: int
    attachment: int
    jealousy: int
    irritation: int
    mood: str
    source_message_count: int
    updated_at: str


@dataclass(slots=True)
class DailyPlan:
    plan_date: str
    summary: str
    schedule: str
    generated_at: str


class HistoryStore:
    def __init__(self, db_path: Path) -> None:
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                direction TEXT NOT NULL,
                chat_label TEXT,
                sender_name TEXT,
                text TEXT NOT NULL,
                local_day TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chat_id, message_id, direction)
            )
            """
        )
        self._ensure_column("messages", "chat_label", "TEXT")
        self._ensure_column("messages", "local_day", "TEXT")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS diary_entries (
                entry_date TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                source_message_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                chat_id INTEGER PRIMARY KEY,
                chat_label TEXT,
                rating INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_memories (
                chat_id INTEGER PRIMARY KEY,
                chat_label TEXT,
                inferred_gender TEXT NOT NULL DEFAULT 'unknown',
                known_facts TEXT NOT NULL DEFAULT '',
                communication_style TEXT NOT NULL DEFAULT '',
                relationship_summary TEXT NOT NULL DEFAULT '',
                recent_events TEXT NOT NULL DEFAULT '',
                affection INTEGER NOT NULL DEFAULT 0,
                trust INTEGER NOT NULL DEFAULT 0,
                loyalty INTEGER NOT NULL DEFAULT 0,
                attachment INTEGER NOT NULL DEFAULT 0,
                jealousy INTEGER NOT NULL DEFAULT 0,
                irritation INTEGER NOT NULL DEFAULT 0,
                mood TEXT NOT NULL DEFAULT 'neutral',
                source_message_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._ensure_column("chat_memories", "affection", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("chat_memories", "trust", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("chat_memories", "loyalty", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("chat_memories", "attachment", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("chat_memories", "jealousy", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("chat_memories", "irritation", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("chat_memories", "mood", "TEXT NOT NULL DEFAULT 'neutral'")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_plans (
                plan_date TEXT PRIMARY KEY,
                summary TEXT NOT NULL DEFAULT '',
                schedule TEXT NOT NULL DEFAULT '',
                generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()

    def _ensure_column(self, table_name: str, column_name: str, column_type: str) -> None:
        rows = self._connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing_columns = {row["name"] for row in rows}
        if column_name in existing_columns:
            return
        self._connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )

    def add_message(
        self,
        *,
        chat_id: int,
        message_id: int,
        direction: str,
        text: str,
        chat_label: str | None = None,
        sender_name: str | None = None,
        created_at: str | None = None,
    ) -> None:
        if created_at:
            created_at_dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        else:
            created_at_dt = datetime.now().astimezone()
        local_day = created_at_dt.date().isoformat()
        created_at_value = created_at or created_at_dt.strftime("%Y-%m-%d %H:%M:%S")
        self._connection.execute(
            """
            INSERT OR IGNORE INTO messages (
                chat_id,
                message_id,
                direction,
                chat_label,
                sender_name,
                text,
                local_day,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                message_id,
                direction,
                chat_label,
                sender_name,
                text,
                local_day,
                created_at_value,
            ),
        )
        self._connection.commit()

    def get_recent_messages(self, chat_id: int, limit: int) -> list[HistoryMessage]:
        rows = self._connection.execute(
            """
            SELECT direction, sender_name, text
            FROM messages
            WHERE chat_id = ? AND direction IN ('incoming', 'outgoing')
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        ).fetchall()

        history: list[HistoryMessage] = []
        for row in reversed(rows):
            if row["direction"] == "outgoing":
                history.append(HistoryMessage(role="assistant", content=row["text"]))
                continue

            sender_name = (row["sender_name"] or "").strip()
            content = row["text"]
            if sender_name:
                content = f"{sender_name}: {content}"
            history.append(HistoryMessage(role="user", content=content))
        return history

    def get_last_incoming_message_id(self, chat_id: int) -> int | None:
        row = self._connection.execute(
            """
            SELECT message_id
            FROM messages
            WHERE chat_id = ? AND direction = 'incoming'
            ORDER BY id DESC
            LIMIT 1
            """,
            (chat_id,),
        ).fetchone()
        if row is None:
            return None
        return int(row["message_id"])

    def get_last_message(self, chat_id: int) -> StoredMessage | None:
        row = self._connection.execute(
            """
            SELECT direction, text, created_at
            FROM messages
            WHERE chat_id = ? AND direction IN ('incoming', 'outgoing')
            ORDER BY id DESC
            LIMIT 1
            """,
            (chat_id,),
        ).fetchone()
        if row is None:
            return None
        return StoredMessage(
            direction=row["direction"],
            text=row["text"],
            created_at=row["created_at"],
        )

    def get_last_outgoing_message(self, chat_id: int) -> StoredMessage | None:
        row = self._connection.execute(
            """
            SELECT direction, text, created_at
            FROM messages
            WHERE chat_id = ? AND direction = 'outgoing'
            ORDER BY id DESC
            LIMIT 1
            """,
            (chat_id,),
        ).fetchone()
        if row is None:
            return None
        return StoredMessage(
            direction=row["direction"],
            text=row["text"],
            created_at=row["created_at"],
        )

    def get_message_count_for_day(self, day_key: str) -> int:
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS message_count
            FROM messages
            WHERE local_day = ?
            """,
            (day_key,),
        ).fetchone()
        if row is None:
            return 0
        return int(row["message_count"])

    def get_day_messages(self, day_key: str, limit: int) -> list[DiarySourceMessage]:
        rows = self._connection.execute(
            """
            SELECT chat_label, direction, sender_name, text, created_at
            FROM messages
            WHERE local_day = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (day_key, limit),
        ).fetchall()

        return [
            DiarySourceMessage(
                chat_label=(row["chat_label"] or "unknown-chat").strip(),
                direction=row["direction"],
                sender_name=row["sender_name"],
                text=row["text"],
                created_at=row["created_at"],
            )
            for row in reversed(rows)
        ]

    def get_message_count_between(self, start_at: str, end_at: str) -> int:
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS message_count
            FROM messages
            WHERE created_at >= ? AND created_at < ?
            """,
            (start_at, end_at),
        ).fetchone()
        if row is None:
            return 0
        return int(row["message_count"])

    def get_messages_between(
        self,
        start_at: str,
        end_at: str,
        limit: int,
    ) -> list[DiarySourceMessage]:
        rows = self._connection.execute(
            """
            SELECT chat_label, direction, sender_name, text, created_at
            FROM messages
            WHERE created_at >= ? AND created_at < ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (start_at, end_at, limit),
        ).fetchall()

        return [
            DiarySourceMessage(
                chat_label=(row["chat_label"] or "unknown-chat").strip(),
                direction=row["direction"],
                sender_name=row["sender_name"],
                text=row["text"],
                created_at=row["created_at"],
            )
            for row in reversed(rows)
        ]

    def get_diary_entry(self, day_key: str) -> DiaryEntry | None:
        row = self._connection.execute(
            """
            SELECT entry_date, summary, source_message_count, updated_at
            FROM diary_entries
            WHERE entry_date = ?
            """,
            (day_key,),
        ).fetchone()
        if row is None:
            return None
        return DiaryEntry(
            entry_date=row["entry_date"],
            summary=row["summary"],
            source_message_count=int(row["source_message_count"]),
            updated_at=row["updated_at"],
        )

    def list_diary_entries_for_day(self, day_key: str) -> list[DiaryEntry]:
        rows = self._connection.execute(
            """
            SELECT entry_date, summary, source_message_count, updated_at
            FROM diary_entries
            WHERE entry_date = ? OR entry_date LIKE ? || 'T%'
            ORDER BY entry_date ASC
            """,
            (day_key, day_key),
        ).fetchall()
        return [
            DiaryEntry(
                entry_date=row["entry_date"],
                summary=row["summary"],
                source_message_count=int(row["source_message_count"]),
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def upsert_diary_entry(
        self,
        *,
        day_key: str,
        summary: str,
        source_message_count: int,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO diary_entries (entry_date, summary, source_message_count, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(entry_date) DO UPDATE SET
                summary = excluded.summary,
                source_message_count = excluded.source_message_count,
                updated_at = CURRENT_TIMESTAMP
            """,
            (day_key, summary, source_message_count),
        )
        self._connection.commit()

    def get_relevant_diary_entries(
        self,
        *,
        query_text: str,
        limit: int,
        lookback_days: int,
    ) -> list[DiaryEntry]:
        rows = self._connection.execute(
            """
            SELECT entry_date, summary, source_message_count, updated_at
            FROM diary_entries
            ORDER BY entry_date DESC
            LIMIT ?
            """,
            (max(lookback_days, limit),),
        ).fetchall()
        if not rows:
            return []

        query_tokens = self._tokenize(query_text)
        scored_entries: list[tuple[int, int, DiaryEntry]] = []
        for index, row in enumerate(rows):
            entry = DiaryEntry(
                entry_date=row["entry_date"],
                summary=row["summary"],
                source_message_count=int(row["source_message_count"]),
                updated_at=row["updated_at"],
            )
            summary_tokens = self._tokenize(entry.summary)
            overlap_score = len(query_tokens.intersection(summary_tokens))
            recency_bonus = max(0, lookback_days - index)
            total_score = overlap_score * 10 + recency_bonus
            scored_entries.append((total_score, -index, entry))

        scored_entries.sort(reverse=True, key=lambda item: (item[0], item[1]))
        selected = [item[2] for item in scored_entries[:limit] if item[0] > 0]
        if selected:
            return selected

        return [
            DiaryEntry(
                entry_date=row["entry_date"],
                summary=row["summary"],
                source_message_count=int(row["source_message_count"]),
                updated_at=row["updated_at"],
            )
            for row in rows[:limit]
        ]

    def get_message_count_for_chat(self, chat_id: int) -> int:
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS message_count
            FROM messages
            WHERE chat_id = ? AND direction IN ('incoming', 'outgoing')
            """,
            (chat_id,),
        ).fetchone()
        if row is None:
            return 0
        return int(row["message_count"])

    def get_chat_memory(self, chat_id: int, chat_label: str | None = None) -> ChatMemory:
        row = self._connection.execute(
            """
            SELECT
                chat_id,
                chat_label,
                inferred_gender,
                known_facts,
                communication_style,
                relationship_summary,
                recent_events,
                affection,
                trust,
                loyalty,
                attachment,
                jealousy,
                irritation,
                mood,
                source_message_count,
                updated_at
            FROM chat_memories
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchone()
        if row is None:
            label = (chat_label or "").strip() or str(chat_id)
            self._connection.execute(
                """
                INSERT INTO chat_memories (
                    chat_id,
                    chat_label,
                    inferred_gender,
                    known_facts,
                    communication_style,
                    relationship_summary,
                    recent_events,
                    affection,
                    trust,
                    loyalty,
                    attachment,
                    jealousy,
                    irritation,
                    mood,
                    source_message_count,
                    updated_at
                ) VALUES (?, ?, 'unknown', '', '', '', '', 0, 0, 0, 0, 0, 0, 'neutral', 0, CURRENT_TIMESTAMP)
                """,
                (chat_id, label),
            )
            self._connection.commit()
            row = self._connection.execute(
                """
                SELECT
                    chat_id,
                    chat_label,
                    inferred_gender,
                    known_facts,
                    communication_style,
                    relationship_summary,
                    recent_events,
                    affection,
                    trust,
                    loyalty,
                    attachment,
                    jealousy,
                    irritation,
                    mood,
                    source_message_count,
                    updated_at
                FROM chat_memories
                WHERE chat_id = ?
                """,
                (chat_id,),
            ).fetchone()
        return ChatMemory(
            chat_id=int(row["chat_id"]),
            chat_label=(row["chat_label"] or "").strip() or (chat_label or str(chat_id)),
            inferred_gender=(row["inferred_gender"] or "unknown").strip() or "unknown",
            known_facts=row["known_facts"] or "",
            communication_style=row["communication_style"] or "",
            relationship_summary=row["relationship_summary"] or "",
            recent_events=row["recent_events"] or "",
            affection=int(row["affection"] or 0),
            trust=int(row["trust"] or 0),
            loyalty=int(row["loyalty"] or 0),
            attachment=int(row["attachment"] or 0),
            jealousy=int(row["jealousy"] or 0),
            irritation=int(row["irritation"] or 0),
            mood=(row["mood"] or "neutral").strip() or "neutral",
            source_message_count=int(row["source_message_count"] or 0),
            updated_at=row["updated_at"],
        )

    def upsert_chat_memory(
        self,
        *,
        chat_id: int,
        chat_label: str,
        inferred_gender: str,
        known_facts: str,
        communication_style: str,
        relationship_summary: str,
        recent_events: str,
        source_message_count: int,
        affection: int | None = None,
        trust: int | None = None,
        loyalty: int | None = None,
        attachment: int | None = None,
        jealousy: int | None = None,
        irritation: int | None = None,
        mood: str | None = None,
    ) -> ChatMemory:
        current = self.get_chat_memory(chat_id, chat_label)
        affection = current.affection if affection is None else self._clamp_emotion(affection)
        trust = current.trust if trust is None else self._clamp_emotion(trust)
        loyalty = current.loyalty if loyalty is None else self._clamp_emotion(loyalty)
        attachment = current.attachment if attachment is None else self._clamp_emotion(attachment)
        jealousy = current.jealousy if jealousy is None else self._clamp_emotion(jealousy)
        irritation = current.irritation if irritation is None else self._clamp_emotion(irritation)
        mood = current.mood if mood is None else (mood.strip() or "neutral")
        self._connection.execute(
            """
            INSERT INTO chat_memories (
                chat_id,
                chat_label,
                inferred_gender,
                known_facts,
                communication_style,
                relationship_summary,
                recent_events,
                affection,
                trust,
                loyalty,
                attachment,
                jealousy,
                irritation,
                mood,
                source_message_count,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chat_id) DO UPDATE SET
                chat_label = excluded.chat_label,
                inferred_gender = excluded.inferred_gender,
                known_facts = excluded.known_facts,
                communication_style = excluded.communication_style,
                relationship_summary = excluded.relationship_summary,
                recent_events = excluded.recent_events,
                affection = excluded.affection,
                trust = excluded.trust,
                loyalty = excluded.loyalty,
                attachment = excluded.attachment,
                jealousy = excluded.jealousy,
                irritation = excluded.irritation,
                mood = excluded.mood,
                source_message_count = excluded.source_message_count,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                chat_id,
                chat_label,
                inferred_gender,
                known_facts,
                communication_style,
                relationship_summary,
                recent_events,
                affection,
                trust,
                loyalty,
                attachment,
                jealousy,
                irritation,
                mood,
                source_message_count,
            ),
        )
        self._connection.commit()
        return self.get_chat_memory(chat_id, chat_label)

    def get_other_chat_memories(
        self,
        *,
        exclude_chat_id: int,
        limit: int,
    ) -> list[ChatMemory]:
        rows = self._connection.execute(
            """
            SELECT
                chat_id,
                chat_label,
                inferred_gender,
                known_facts,
                communication_style,
                relationship_summary,
                recent_events,
                affection,
                trust,
                loyalty,
                attachment,
                jealousy,
                irritation,
                mood,
                source_message_count,
                updated_at
            FROM chat_memories
            WHERE chat_id != ?
              AND (
                known_facts != ''
                OR relationship_summary != ''
                OR recent_events != ''
                OR source_message_count > 0
              )
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (exclude_chat_id, limit),
        ).fetchall()
        return [
            ChatMemory(
                chat_id=int(row["chat_id"]),
                chat_label=(row["chat_label"] or "").strip() or str(row["chat_id"]),
                inferred_gender=(row["inferred_gender"] or "unknown").strip() or "unknown",
                known_facts=row["known_facts"] or "",
                communication_style=row["communication_style"] or "",
                relationship_summary=row["relationship_summary"] or "",
                recent_events=row["recent_events"] or "",
                affection=int(row["affection"] or 0),
                trust=int(row["trust"] or 0),
                loyalty=int(row["loyalty"] or 0),
                attachment=int(row["attachment"] or 0),
                jealousy=int(row["jealousy"] or 0),
                irritation=int(row["irritation"] or 0),
                mood=(row["mood"] or "neutral").strip() or "neutral",
                source_message_count=int(row["source_message_count"] or 0),
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_daily_plan(self, plan_date: str) -> DailyPlan | None:
        row = self._connection.execute(
            """
            SELECT plan_date, summary, schedule, generated_at
            FROM daily_plans
            WHERE plan_date = ?
            """,
            (plan_date,),
        ).fetchone()
        if row is None:
            return None
        return DailyPlan(
            plan_date=row["plan_date"],
            summary=row["summary"] or "",
            schedule=row["schedule"] or "",
            generated_at=row["generated_at"],
        )

    def upsert_daily_plan(
        self,
        *,
        plan_date: str,
        summary: str,
        schedule: str,
    ) -> DailyPlan:
        self._connection.execute(
            """
            INSERT INTO daily_plans (plan_date, summary, schedule, generated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(plan_date) DO UPDATE SET
                summary = excluded.summary,
                schedule = excluded.schedule,
                generated_at = CURRENT_TIMESTAMP
            """,
            (plan_date, summary, schedule),
        )
        self._connection.commit()
        plan = self.get_daily_plan(plan_date)
        assert plan is not None
        return plan

    def update_chat_emotions(
        self,
        *,
        chat_id: int,
        chat_label: str,
        affection_delta: int = 0,
        trust_delta: int = 0,
        loyalty_delta: int = 0,
        attachment_delta: int = 0,
        jealousy_delta: int = 0,
        irritation_delta: int = 0,
        mood: str | None = None,
    ) -> ChatMemory:
        current = self.get_chat_memory(chat_id, chat_label)
        new_mood = mood.strip() if mood and mood.strip() else self._derive_mood(
            affection=self._clamp_emotion(current.affection + affection_delta),
            trust=self._clamp_emotion(current.trust + trust_delta),
            loyalty=self._clamp_emotion(current.loyalty + loyalty_delta),
            attachment=self._clamp_emotion(current.attachment + attachment_delta),
            jealousy=self._clamp_emotion(current.jealousy + jealousy_delta),
            irritation=self._clamp_emotion(current.irritation + irritation_delta),
        )
        self._connection.execute(
            """
            INSERT INTO chat_memories (
                chat_id,
                chat_label,
                inferred_gender,
                known_facts,
                communication_style,
                relationship_summary,
                recent_events,
                affection,
                trust,
                loyalty,
                attachment,
                jealousy,
                irritation,
                mood,
                source_message_count,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chat_id) DO UPDATE SET
                chat_label = excluded.chat_label,
                affection = excluded.affection,
                trust = excluded.trust,
                loyalty = excluded.loyalty,
                attachment = excluded.attachment,
                jealousy = excluded.jealousy,
                irritation = excluded.irritation,
                mood = excluded.mood,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                chat_id,
                chat_label,
                current.inferred_gender,
                current.known_facts,
                current.communication_style,
                current.relationship_summary,
                current.recent_events,
                self._clamp_emotion(current.affection + affection_delta),
                self._clamp_emotion(current.trust + trust_delta),
                self._clamp_emotion(current.loyalty + loyalty_delta),
                self._clamp_emotion(current.attachment + attachment_delta),
                self._clamp_emotion(current.jealousy + jealousy_delta),
                self._clamp_emotion(current.irritation + irritation_delta),
                new_mood,
                current.source_message_count,
            ),
        )
        self._connection.commit()
        return self.get_chat_memory(chat_id, chat_label)

    @staticmethod
    def _clamp_emotion(value: int) -> int:
        return max(0, min(100, int(value)))

    @staticmethod
    def _derive_mood(
        *,
        affection: int,
        trust: int,
        loyalty: int,
        attachment: int,
        jealousy: int,
        irritation: int,
    ) -> str:
        if irritation >= 65:
            return "hurt / guarded"
        if jealousy >= 55 and affection >= 45:
            return "jealous / attached"
        if affection >= 75 and trust >= 55:
            return "loving / soft"
        if attachment >= 70 or loyalty >= 70:
            return "attached / loyal"
        if trust >= 55:
            return "comfortable"
        if irritation >= 35:
            return "annoyed"
        return "neutral"

    def get_user_profile(self, chat_id: int, chat_label: str | None = None) -> UserProfile:
        row = self._connection.execute(
            """
            SELECT chat_id, chat_label, rating, notes, updated_at
            FROM user_profiles
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchone()
        if row is None:
            label = (chat_label or "").strip() or str(chat_id)
            self._connection.execute(
                """
                INSERT INTO user_profiles (chat_id, chat_label, rating, notes, updated_at)
                VALUES (?, ?, 0, '', CURRENT_TIMESTAMP)
                """,
                (chat_id, label),
            )
            self._connection.commit()
            row = self._connection.execute(
                """
                SELECT chat_id, chat_label, rating, notes, updated_at
                FROM user_profiles
                WHERE chat_id = ?
                """,
                (chat_id,),
            ).fetchone()
        return UserProfile(
            chat_id=int(row["chat_id"]),
            chat_label=(row["chat_label"] or "").strip() or (chat_label or str(chat_id)),
            rating=int(row["rating"]),
            notes=row["notes"] or "",
            updated_at=row["updated_at"],
        )

    def update_user_profile(
        self,
        *,
        chat_id: int,
        chat_label: str,
        rating_delta: int = 0,
        notes: str | None = None,
    ) -> UserProfile:
        profile = self.get_user_profile(chat_id, chat_label)
        new_rating = max(-100, min(100, profile.rating + rating_delta))
        new_notes = notes if notes is not None else profile.notes
        self._connection.execute(
            """
            INSERT INTO user_profiles (chat_id, chat_label, rating, notes, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chat_id) DO UPDATE SET
                chat_label = excluded.chat_label,
                rating = excluded.rating,
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
            """,
            (chat_id, chat_label, new_rating, new_notes),
        )
        self._connection.commit()
        return self.get_user_profile(chat_id, chat_label)

    def set_user_profile(
        self,
        *,
        chat_id: int,
        chat_label: str,
        rating: int,
        notes: str | None = None,
    ) -> UserProfile:
        profile = self.get_user_profile(chat_id, chat_label)
        normalized_rating = max(-100, min(100, rating))
        new_notes = notes if notes is not None else profile.notes
        self._connection.execute(
            """
            INSERT INTO user_profiles (chat_id, chat_label, rating, notes, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chat_id) DO UPDATE SET
                chat_label = excluded.chat_label,
                rating = excluded.rating,
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
            """,
            (chat_id, chat_label, normalized_rating, new_notes),
        )
        self._connection.commit()
        return self.get_user_profile(chat_id, chat_label)

    def get_message_direction(self, chat_id: int, message_id: int) -> str | None:
        row = self._connection.execute(
            """
            SELECT direction
            FROM messages
            WHERE chat_id = ? AND message_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (chat_id, message_id),
        ).fetchone()
        if row is None:
            return None
        return row["direction"]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = re.findall(r"[A-Za-z\u0400-\u04FF0-9_]{3,}", text.lower())
        return set(tokens)
