"""
CareerMate - Conversation Session Manager
Manages conversation sessions, chat histories, active documents, and persistence.
"""
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import Config

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages multi-chat conversations, history retrieval, and session-level document bindings.
    Persists data to JSON so chat histories are maintained across sessions.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (Config.BASE_DIR / "data" / "sessions.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._load_sessions()

    def _load_sessions(self):
        """Loads sessions from local JSON storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.sessions = json.load(f)
                logger.info(f"Loaded {len(self.sessions)} sessions from storage.")
            except Exception as e:
                logger.error(f"Error loading sessions: {e}. Initializing fresh session store.")
                self.sessions = {}
        else:
            self.sessions = {}

    def _save_sessions(self):
        """Saves current sessions to local JSON storage."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")

    def create_session(self, title: str = "New Career Chat") -> Dict[str, Any]:
        """Creates a brand new conversation session."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        session_data = {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "uploaded_document": None
        }
        self.sessions[session_id] = session_data
        self._save_sessions()
        return session_data

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a session by its unique ID. Creates one if non-existent."""
        if not session_id:
            return self.create_session()
        if session_id not in self.sessions:
            session_data = self.create_session()
            return session_data
        return self.sessions[session_id]

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Returns sorted list of sessions (most recently updated first)."""
        session_list = []
        for s_id, s_data in self.sessions.items():
            session_list.append({
                "id": s_id,
                "title": s_data.get("title", "New Career Chat"),
                "created_at": s_data.get("created_at"),
                "updated_at": s_data.get("updated_at"),
                "message_count": len(s_data.get("messages", [])),
                "uploaded_document": s_data.get("uploaded_document")
            })
        session_list.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return session_list

    def add_message(
        self,
        session_id: str,
        sender: str,
        text: str,
        citations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Adds a message to the specified session."""
        session = self.get_session(session_id)
        msg_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        message = {
            "id": msg_id,
            "sender": sender,  # "user" or "ai"
            "text": text,
            "timestamp": now,
            "citations": citations or []
        }
        session["messages"].append(message)
        session["updated_at"] = now

        # Auto-update session title based on the first user query
        if sender == "user" and session.get("title") == "New Career Chat":
            clean_title = text.strip()
            if len(clean_title) > 35:
                clean_title = clean_title[:32] + "..."
            session["title"] = clean_title

        self._save_sessions()
        return message

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns all messages in the specified session."""
        session = self.get_session(session_id)
        return session.get("messages", [])

    def clear_session_messages(self, session_id: str):
        """Clears all conversation messages from the session."""
        if session_id in self.sessions:
            self.sessions[session_id]["messages"] = []
            self.sessions[session_id]["updated_at"] = datetime.utcnow().isoformat()
            self._save_sessions()

    def set_session_document(self, session_id: str, filename: Optional[str]):
        """Binds or clears an uploaded PDF document for a session."""
        session = self.get_session(session_id)
        session["uploaded_document"] = filename
        session["updated_at"] = datetime.utcnow().isoformat()
        self._save_sessions()

    def get_session_document(self, session_id: str) -> Optional[str]:
        """Gets the active uploaded document name for a session."""
        session = self.get_session(session_id)
        return session.get("uploaded_document")

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session permanently."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
            return True
        return False
