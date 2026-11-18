"""Conversations and messages in the database, and one chat turn: user message in, answer out, both stored.

`run_turn` is blocking (database + knowledge base + DeepSeek), so the router runs it on a worker thread. It does ALL the
persistence itself: if the browser disconnects half-way the answer is still saved and shows up when the chat is reopened.
"""
from __future__ import annotations

import logging
from typing import Callable

from psycopg.types.json import Jsonb

from .errors import ApiError, not_found
from .state import AppState
from .users_repo import engine_profile, iso

log = logging.getLogger("legalmate.chat")

TAG_ICON = {"Property Law": "home", "Family Law": "person", "Criminal Law": "doc", "Labour & Employment": "briefcase",
            "Contract Law": "file", "Consumer Rights": "message", "Constitutional Rights": "doc", "Civil Procedure": "file",
            "Cyber Law": "doc"}
# Starter cards of the chat screen: title and legal area are known without asking the model.
TOPICS = {"tenant-rights": ("Tenant rights in Pakistan", "Property Law"), "employment": ("Employment termination", "Labour & Employment"),
          "inheritance": ("Family inheritance", "Family Law"), "consumer": ("Consumer complaint", "Consumer Rights")}
CONV_COLS = "id, title, title_auto, tag, icon, preview, pinned, created_at, updated_at"


def conv_out(r: dict) -> dict:
    return {"id": str(r["id"]), "title": r["title"], "tag": r["tag"], "icon": r["icon"], "preview": r["preview"],
            "pinned": r["pinned"], "updatedAt": iso(r["updated_at"])}


def attachment_names(state: AppState, rows: list[dict]) -> dict[str, str]:
    """File names for the attachments in these messages (a document that has expired or was deleted has no name any more)."""
    ids = sorted({str(a) for r in rows for a in (r.get("attachment_ids") or [])})
    if not ids:
        return {}
    return {str(d["id"]): d["name"] for d in state.db.all("select id, name from app.documents where id = any(%s::uuid[])", [ids])}


def msg_out(r: dict, names: dict[str, str] | None = None) -> dict:
    if r["role"] == "user":
        ids = [str(a) for a in (r.get("attachment_ids") or [])]
        out = {"id": str(r["id"]), "role": "user", "kind": "user", "content": r["content"], "attachmentIds": ids, "createdAt": iso(r["created_at"])}
        if ids:
            out["attachments"] = [{"id": i, "name": (names or {}).get(i, "Attached file")} for i in ids]
        return out
    return {**(r["payload"] or {}), "id": str(r["id"]), "role": "assistant", "kind": r["kind"], "createdAt": iso(r["created_at"]),
            "feedback": r["feedback"]}


def get_conversation(state: AppState, user_id: str, conv_id: str) -> dict:
    row = state.db.one(f"select {CONV_COLS} from app.conversations where id = %s and user_id = %s", [conv_id, user_id])
    if row is None:
        raise not_found("That conversation")
    return row


def create_conversation(state: AppState, user_id: str, first_message: str, topic: str | None) -> dict:
    title, tag = TOPICS.get(topic or "", (None, "General"))
    if title is None:
        title = " ".join(first_message.split())[:48] or "New conversation"
    return state.db.one(
        f"insert into app.conversations (user_id, title, tag, icon, preview) values (%s, %s, %s, %s, %s) returning {CONV_COLS}",
        [user_id, title, tag, TAG_ICON.get(tag, "chat"), " ".join(first_message.split())[:60]])


def _flat(parts) -> str:
    """Rich text ([text, {b}, {cite}]) as plain words: bold kept as words, citation chips dropped (their numbers only mean something inside one answer)."""
    if isinstance(parts, str):
        return parts
    return "".join(p if isinstance(p, str) else p.get("b", "") for p in (parts or []) if isinstance(p, (str, dict)))


def history_text(r: dict) -> str:
    """One stored message as the model should read it: an answer in FULL (what it assumed, what the law says, the steps, the checklist, the warning),
    a clarifying question with its options, anything else as it was written. Nothing is cut."""
    p = r.get("payload") or {}
    if r["kind"] == "answer" and p:
        lines = [("Assumed: " + p["assumptions"]) if p.get("assumptions") else "", _flat(p.get("intro")), _flat(p.get("law"))]
        lines += [f"Step {i}: {_flat(s)}" for i, s in enumerate(p.get("steps") or [], start=1)]
        if p.get("prepare"):
            lines.append("Keep ready: " + "; ".join(str(x) for x in p["prepare"]))
        lines += [("Important: " + p["important"]) if p.get("important") else "", p.get("outro") or ""]
        text = "\n".join(x for x in lines if x and x.strip())
        return text or r["content"]
    if r["kind"] == "clarify" and p.get("questions"):
        return (p.get("intro") or "") + " " + " ".join(f"{q['text']} (options: {' / '.join(q.get('options') or [])})" for q in p["questions"])
    return r["content"]


def _history(state: AppState, conv_id: str) -> list[dict]:
    """EVERY earlier message of this conversation, oldest first, uncut. The model is given the whole chat, however long it is."""
    rows = state.db.all("select role, kind, content, payload from app.messages where conversation_id = %s order by seq asc", [conv_id])
    return [{"role": r["role"], "kind": r["kind"], "content": history_text(r)} for r in rows]


def _safe_meta(meta: dict) -> dict:
    """Diagnostics worth keeping (timings, verification results, what was retrieved with). Never the raw user text."""
    keep = ("analysis", "language", "language_mismatch", "verify", "retrieval_seconds", "seconds", "llm_calls_this_turn", "pii_removed", "error")
    return {k: meta[k] for k in keep if k in meta}


def run_turn(state: AppState, user: dict, conv_id: str, text: str, language: str, on_status: Callable[[str], None], docs: list[dict] | None = None) -> dict:
    """Returns {"conversation": ..., "message": ...} for the browser, or raises ApiError (the router turns it into an `error` event)."""
    uid = str(user["id"])
    user_msg = None
    try:
        history = _history(state, conv_id)
        if not state.engine.ready and state.engine.enabled:
            on_status("loading_library")                                         # the library is still loading after a restart: say so instead of a silent wait
        engine = state.engine.get()
        conv = get_conversation(state, uid, conv_id)
        user_msg = state.db.one("insert into app.messages (conversation_id, user_id, role, kind, content, attachment_ids) "
                                "values (%s, %s, 'user', 'user', %s, %s::uuid[]) returning id",
                                [conv_id, uid, text, [str(d["id"]) for d in (docs or [])]])
        doc_text = None
        if docs:                                           # the part of each attached document that matters for this question
            from .docs_service import read_pending, select_context
            pending = [d for d in docs if d.get("status") == "pending"]
            if pending:                                    # photos are read now, not at upload: the chat shows "Analyzing the image…"
                on_status("reading_image")
                for d in pending:
                    read_pending(state, uid, d)
            doc_text = select_context(docs, text, state.settings.docs.chat_doc_chars)
        out = engine.respond(text, history=history, profile=engine_profile(user), language=language, on_status=on_status, attachment_text=doc_text)
        msg, meta = out["message"], out.get("meta", {})

        if msg["kind"] == "topic_change":                       # a different matter: not answered here, and the question does not stay in this chat
            state.db.run("delete from app.messages where id = %s", [user_msg["id"]])
            user_msg = None
            return {"topicChange": {"text": msg["text"], "title": msg.get("title"), "tag": msg.get("tag"), "seconds": msg["seconds"], "language": msg.get("language")}}

        if msg["kind"] == "info" and msg.get("reason") in ("error", "busy"):      # nothing useful was produced
            raise ApiError(503 if msg["reason"] == "error" else 429, "UNAVAILABLE" if msg["reason"] == "error" else "RATE_LIMITED", msg["text"])

        payload = {k: v for k, v in msg.items() if k not in ("createdAt", "feedback")}
        analysed = out.get("conversation") if conv["title_auto"] else None
        with state.db.tx() as c:
            row = c.execute(
                "insert into app.messages (conversation_id, user_id, role, kind, content, payload, meta) "
                "values (%s, %s, 'assistant', %s, %s, %s, %s) returning id, seq, role, kind, content, payload, attachment_ids, feedback, created_at",
                [conv_id, uid, msg["kind"], msg.get("content") or msg.get("text") or "", Jsonb(payload), Jsonb(_safe_meta(meta))]).fetchone()
            title, tag = (analysed["title"], analysed["tag"]) if analysed else (conv["title"], conv["tag"])
            new_conv = c.execute(
                f"update app.conversations set title = %s, tag = %s, icon = %s, title_auto = %s, preview = %s, updated_at = now() "
                f"where id = %s returning {CONV_COLS}",
                [title, tag, TAG_ICON.get(tag, conv["icon"]), conv["title_auto"] and not analysed, " ".join(text.split())[:60], conv_id]).fetchone()
        return {"conversation": conv_out(new_conv), "message": msg_out(row)}
    except Exception as e:
        if user_msg is not None:                                  # leave no half-turn behind: the question without an answer
            try:
                state.db.run("delete from app.messages where id = %s", [user_msg["id"]])
            except Exception:                                     # noqa: BLE001
                log.exception("could not remove the unanswered message")
        if isinstance(e, ApiError):
            raise
        log.exception("chat turn failed (conversation %s)", conv_id)
        raise ApiError(500, "INTERNAL", "LegalMate could not answer right now. Please try again.")
