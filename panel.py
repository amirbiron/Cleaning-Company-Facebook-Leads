"""פאנל הגדרות — שרת Flask לניהול הגדרות הבוט דרך ממשק ווב."""

import hmac
import os
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from database import (
    init_db,
    get_config,
    set_config,
    get_config_encrypted,
    set_config_encrypted,
    add_group,
    remove_group,
    get_db_groups,
    count_groups,
    _get_max_groups,
    add_keyword,
    remove_keyword,
    get_db_keywords,
    ensure_keywords_migrated,
    add_blocked_user,
    remove_blocked_user,
    get_blocked_users,
    get_all_group_health,
)
from logger import get_logger

log = get_logger("Panel")

PANEL_PORT = int(os.environ.get("PANEL_PORT", os.environ.get("PORT", "8080")))


def _safe_int(value, default: int) -> int:
    """המרה בטוחה ל-int עם fallback — מונע crash כש-DB מכיל ערך לא תקין."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


class _QuietRequestFilter:
    """מסנן בקשות polling תכופות מלוגים של werkzeug — מונע רעש."""
    _QUIET_PATHS = {"/api/scan-status"}

    def filter(self, record):
        msg = record.getMessage() if hasattr(record, 'getMessage') else str(record.msg)
        # werkzeug מלוגג בפורמט: '10.0.0.1 - - [date] "GET /path ..." 200 -'
        for path in self._QUIET_PATHS:
            if f'"GET {path} ' in msg or f'"GET {path}?' in msg:
                return False
        return True


def create_app() -> Flask:
    app = Flask(__name__)

    # הגבלת קצב בקשות — מונע brute-force על PANEL_TOKEN והצפת שרת
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["60 per minute"],
        storage_uri="memory://",
    )

    # השתקת לוגי werkzeug לבקשות polling תכופות (scan-status כל 3 שניות)
    import logging as _logging
    werkzeug_log = _logging.getLogger("werkzeug")
    werkzeug_log.addFilter(_QuietRequestFilter())

    # ── אימות — PANEL_TOKEN ──────────────────────────────────
    # אם PANEL_TOKEN מוגדר, כל בקשה ל-API חייבת לכלול את הטוקן
    # ב-header: Authorization: Bearer <token>
    # דף ה-HTML עצמו נגיש בלי טוקן — הוא שולח את הטוקן מ-localStorage
    panel_token = os.environ.get("PANEL_TOKEN", "").strip()
    if not panel_token:
        log.warning("PANEL_TOKEN לא מוגדר — הפאנל פתוח ללא אימות!")

    _expected_auth = f"Bearer {panel_token}" if panel_token else ""

    def require_auth(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not panel_token:
                return f(*args, **kwargs)
            auth = request.headers.get("Authorization", "")
            if hmac.compare_digest(auth, _expected_auth):
                return f(*args, **kwargs)
            return jsonify({"ok": False, "message": "אימות נדרש"}), 401
        return wrapper

    # ── דף ראשי (ללא אימות — ה-HTML עצמו לא מכיל מידע רגיש) ──
    @app.route("/")
    def index():
        return send_file(Path(__file__).parent / "panel.html")

    @app.route("/api/auth", methods=["POST"])
    @limiter.limit("10 per minute")
    def check_auth():
        """בדיקת תקינות טוקן. מחזיר ok=true אם הטוקן נכון או אם אין צורך באימות."""
        if not panel_token:
            return jsonify({"ok": True, "auth_required": False})
        auth = request.headers.get("Authorization", "")
        if hmac.compare_digest(auth, _expected_auth):
            return jsonify({"ok": True, "auth_required": True})
        return jsonify({"ok": False, "auth_required": True}), 401

    # ── הגדרות כלליות ─────────────────────────────────────────

    # sentinel — מבדיל בין "אין ערך ב-DB" ל-"ערך ריק ב-DB"
    _NOT_SET = object()

    @app.route("/api/settings", methods=["GET"])
    @require_auth
    def get_settings():
        from classifier import _CLASSIFICATION_CRITERIA_DEFAULT
        import hashlib

        criteria_db = get_config("classification_criteria")
        is_default = not bool(criteria_db.strip()) if criteria_db else True

        # בדיקה אם ברירת המחדל השתנתה מאז שהמשתמש שמר פרומפט מותאם
        criteria_default_changed = False
        if not is_default:
            saved_hash = get_config("criteria_default_hash_at_save")
            current_hash = hashlib.md5(
                _CLASSIFICATION_CRITERIA_DEFAULT.strip().encode()
            ).hexdigest()
            if not saved_hash or saved_hash != current_hash:
                criteria_default_changed = True

        # quiet_hours — ערך ריק ב-DB = המשתמש ביטל דרך הפאנל → לא לחזור ל-env
        qh_db = get_config("quiet_hours", _NOT_SET)
        quiet_hours = qh_db if qh_db is not _NOT_SET else os.environ.get("QUIET_HOURS", "")

        # interval_minutes — env var בלבד (הוסר מהפאנל)
        interval = os.environ.get("INTERVAL_MINUTES", "10")

        # max_post_age_days — 0 או ריק = כבוי
        max_age_raw = get_config("max_post_age_days")
        max_post_age_days = 0
        if max_age_raw:
            try:
                max_post_age_days = int(max_age_raw)
            except (ValueError, TypeError):
                pass

        # סיסמה — לא חושפים את הערך, רק מצב set/unset
        fb_password_raw = get_config_encrypted("fb_password") or os.environ.get("FB_PASSWORD", "")
        return jsonify({
            "fb_email": get_config("fb_email") or os.environ.get("FB_EMAIL", ""),
            "fb_password_set": bool(fb_password_raw),
            "classification_criteria": criteria_db or _CLASSIFICATION_CRITERIA_DEFAULT,
            "classification_criteria_default": _CLASSIFICATION_CRITERIA_DEFAULT,
            "criteria_is_default": is_default,
            "criteria_default_changed": criteria_default_changed,
            "quiet_hours": quiet_hours,
            "interval_minutes": interval,
            "max_post_age_days": max_post_age_days,
            "inactive_group_threshold": _safe_int(get_config("inactive_group_threshold", "50"), 50),
        })

    @app.route("/api/settings", methods=["PUT"])
    @require_auth
    def update_settings():
        data = request.get_json(force=True)
        allowed = ("fb_email", "fb_password", "classification_criteria",
                    "quiet_hours", "max_post_age_days", "inactive_group_threshold")
        for key in allowed:
            if key in data:
                value = str(data[key]).strip()
                # סיסמה ריקה = הפאנל לא שינה אותה (השדה מוחזר ריק ב-GET) — לא לדרוס
                if key == "fb_password":
                    if value:
                        set_config_encrypted(key, value)
                elif key == "inactive_group_threshold":
                    # ולידציה — חייב להיות מספר שלם >= 1
                    try:
                        val_int = int(value)
                        if val_int >= 1:
                            set_config(key, str(val_int))
                    except (ValueError, TypeError):
                        pass
                else:
                    set_config(key, value)

        # שמירת hash של ברירת המחדל בזמן שמירת פרומפט מותאם
        if "classification_criteria" in data:
            import hashlib
            val = str(data["classification_criteria"]).strip()
            if val:
                from classifier import _CLASSIFICATION_CRITERIA_DEFAULT
                h = hashlib.md5(
                    _CLASSIFICATION_CRITERIA_DEFAULT.strip().encode()
                ).hexdigest()
                set_config("criteria_default_hash_at_save", h)
            else:
                # איפוס לברירת מחדל — מנקים את ה-hash
                set_config("criteria_default_hash_at_save", "")

        return jsonify({"ok": True})

    # ── קבוצות ────────────────────────────────────────────────

    @app.route("/api/groups", methods=["GET"])
    @require_auth
    def list_groups():
        groups = get_db_groups()
        max_groups = _get_max_groups()
        result = {
            "groups": groups or [],
            "source": "none" if groups is None else "custom",
            "max_groups": max_groups,   # 0 = ללא הגבלה
            "count": count_groups(),
        }
        return jsonify(result)

    @app.route("/api/groups", methods=["POST"])
    @require_auth
    def create_group():
        data = request.get_json(force=True)
        url = data.get("url", "").strip()
        if not url:
            return jsonify({"ok": False, "message": "URL ריק"}), 400
        ok, msg = add_group(url)
        return jsonify({"ok": ok, "message": msg})

    @app.route("/api/groups", methods=["DELETE"])
    @require_auth
    def delete_group():
        data = request.get_json(force=True)
        url = data.get("url", "").strip()
        if not url:
            return jsonify({"ok": False, "message": "URL ריק"}), 400
        ok, msg = remove_group(url)
        return jsonify({"ok": ok, "message": msg})

    # ── מילות "שלח תמיד" (force_send) ────────────────────────

    @app.route("/api/force_send", methods=["GET"])
    @require_auth
    def list_force_send():
        from main import _load_force_send_keywords
        keywords = _load_force_send_keywords()
        return jsonify({"keywords": keywords})

    @app.route("/api/force_send", methods=["POST"])
    @require_auth
    def create_force_send():
        data = request.get_json(force=True)
        word = data.get("word", "").strip()
        if not word:
            return jsonify({"ok": False, "message": "מילה ריקה"}), 400
        from main import add_force_send_keyword
        ok, msg = add_force_send_keyword(word)
        return jsonify({"ok": ok, "message": msg})

    @app.route("/api/force_send", methods=["DELETE"])
    @require_auth
    def delete_force_send():
        data = request.get_json(force=True)
        word = data.get("word", "").strip()
        if not word:
            return jsonify({"ok": False, "message": "מילה ריקה"}), 400
        from main import remove_force_send_keyword
        ok, msg = remove_force_send_keyword(word)
        return jsonify({"ok": ok, "message": msg})

    # ── מילות "שלח תמיד" לקבוצה ספציפית ──────────────────────

    @app.route("/api/group_force_send", methods=["GET"])
    @require_auth
    def list_all_group_force_send():
        """מחזיר את כל מילות force_send לפי קבוצה."""
        from main import get_all_group_force_send
        return jsonify({"groups": get_all_group_force_send()})

    @app.route("/api/group_force_send/keywords", methods=["GET"])
    @require_auth
    def list_group_force_send():
        """מחזיר מילות force_send לקבוצה ספציפית."""
        group_url = request.args.get("url", "").strip()
        if not group_url:
            return jsonify({"ok": False, "message": "URL ריק"}), 400
        from main import _load_group_force_send_keywords
        keywords = _load_group_force_send_keywords(group_url)
        return jsonify({"keywords": keywords})

    @app.route("/api/group_force_send/keywords", methods=["POST"])
    @require_auth
    def create_group_force_send():
        """מוסיף מילת force_send לקבוצה ספציפית."""
        data = request.get_json(force=True)
        group_url = data.get("url", "").strip()
        word = data.get("word", "").strip()
        if not group_url:
            return jsonify({"ok": False, "message": "URL ריק"}), 400
        if not word:
            return jsonify({"ok": False, "message": "מילה ריקה"}), 400
        from main import add_group_force_send_keyword
        ok, msg = add_group_force_send_keyword(group_url, word)
        return jsonify({"ok": ok, "message": msg})

    @app.route("/api/group_force_send/keywords", methods=["DELETE"])
    @require_auth
    def delete_group_force_send():
        """מסיר מילת force_send מקבוצה ספציפית."""
        data = request.get_json(force=True)
        group_url = data.get("url", "").strip()
        word = data.get("word", "").strip()
        if not group_url:
            return jsonify({"ok": False, "message": "URL ריק"}), 400
        if not word:
            return jsonify({"ok": False, "message": "מילה ריקה"}), 400
        from main import remove_group_force_send_keyword
        ok, msg = remove_group_force_send_keyword(group_url, word)
        return jsonify({"ok": ok, "message": msg})

    # ── סטטוס סריקה בזמן אמת ────────────────────────────────

    @app.route("/api/scan-status", methods=["GET"])
    @require_auth
    def get_scan_status():
        from main import scan_progress
        from database import get_stats, get_daily_stats
        from datetime import datetime
        import os
        tz_name = os.environ.get("TIMEZONE", "UTC")
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(tz_name)
        except Exception:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo("UTC")
        now = datetime.now(tz=tz)
        today_prefix = now.strftime("%Y-%m-%d")
        daily = get_daily_stats(today_prefix)
        total = get_stats()
        # העתקה עמוקה של groups_done — מונע race condition עם _on_group_scraped
        # שמוסיף לרשימה מ-thread אחר בזמן הסריאליזציה
        import copy
        progress_snapshot = copy.deepcopy(scan_progress)
        return jsonify({
            **progress_snapshot,
            "daily_seen": daily["seen"],
            "daily_sent": daily["sent"],
            "total_seen": total["seen"],
            "total_sent": total["sent"],
            "server_time": now.strftime("%H:%M:%S"),
        })

    # ── בריאות קבוצות ───────────────────────────────────────────

    @app.route("/api/group-health", methods=["GET"])
    @require_auth
    def get_group_health():
        health = get_all_group_health()
        groups = get_db_groups() or []
        url_to_name = {g["url"]: g["name"] for g in groups}
        # מוסיפים שם ידידותי לכל רשומה
        for h in health:
            h["group_name"] = url_to_name.get(
                h["group_url"],
                h["group_url"].split("/groups/")[-1].rstrip("/") or h["group_url"],
            )
        threshold = _safe_int(get_config("inactive_group_threshold", "50"), 50)
        return jsonify({"health": health, "inactive_threshold": threshold})

    @app.route("/api/group-health/settings", methods=["PUT"])
    @require_auth
    def update_group_health_settings():
        data = request.get_json(force=True)
        threshold = data.get("inactive_threshold")
        if threshold is not None:
            try:
                val = int(threshold)
                if val < 1:
                    return jsonify({"ok": False, "message": "הסף חייב להיות לפחות 1"}), 400
                set_config("inactive_group_threshold", str(val))
                return jsonify({"ok": True, "inactive_threshold": val})
            except (ValueError, TypeError):
                return jsonify({"ok": False, "message": "ערך לא תקין"}), 400
        return jsonify({"ok": False, "message": "חסר inactive_threshold"}), 400

    # ── מילות מפתח ────────────────────────────────────────────

    @app.route("/api/keywords/<kw_type>", methods=["GET"])
    @require_auth
    def list_keywords(kw_type):
        if kw_type not in ("pre_filter", "block"):
            return jsonify({"ok": False, "message": "סוג לא תקין"}), 400

        kws = get_db_keywords(kw_type)
        if kws is None:
            # fallback לברירות מחדל
            if kw_type == "pre_filter":
                from main import _PRE_FILTER_KEYWORDS_DEFAULT
                return jsonify({"keywords": _PRE_FILTER_KEYWORDS_DEFAULT, "source": "default"})
            else:
                from main import _BLOCK_KEYWORDS_ENV
                return jsonify({"keywords": _BLOCK_KEYWORDS_ENV, "source": "default"})
        return jsonify({"keywords": kws, "source": "custom"})

    @app.route("/api/keywords/<kw_type>", methods=["POST"])
    @require_auth
    def create_keyword(kw_type):
        if kw_type not in ("pre_filter", "block"):
            return jsonify({"ok": False, "message": "סוג לא תקין"}), 400
        data = request.get_json(force=True)
        word = data.get("word", "").strip()
        if not word:
            return jsonify({"ok": False, "message": "מילה ריקה"}), 400
        # מיגרציית ברירות מחדל — אם זו ההוספה הראשונה, מעביר defaults ל-DB
        if kw_type == "pre_filter":
            from main import _PRE_FILTER_KEYWORDS_DEFAULT
            ensure_keywords_migrated(kw_type, _PRE_FILTER_KEYWORDS_DEFAULT)
        elif kw_type == "block":
            from main import _BLOCK_KEYWORDS_ENV
            ensure_keywords_migrated(kw_type, _BLOCK_KEYWORDS_ENV)
        ok, msg = add_keyword(word, kw_type)
        return jsonify({"ok": ok, "message": msg})

    @app.route("/api/keywords/<kw_type>", methods=["DELETE"])
    @require_auth
    def delete_keyword(kw_type):
        if kw_type not in ("pre_filter", "block"):
            return jsonify({"ok": False, "message": "סוג לא תקין"}), 400
        data = request.get_json(force=True)
        word = data.get("word", "").strip()
        if not word:
            return jsonify({"ok": False, "message": "מילה ריקה"}), 400
        ok, msg = remove_keyword(word, kw_type)
        return jsonify({"ok": ok, "message": msg})

    # ── מפרסמים חסומים (blocked_users) ───────────────────────

    @app.route("/api/blocked_users", methods=["GET"])
    @require_auth
    def list_blocked_users():
        users = get_blocked_users()
        return jsonify({"users": users})

    @app.route("/api/blocked_users", methods=["POST"])
    @require_auth
    def create_blocked_user():
        data = request.get_json(force=True)
        url = data.get("url", "").strip()
        if not url:
            return jsonify({"ok": False, "message": "URL ריק"}), 400
        ok, msg = add_blocked_user(url)
        return jsonify({"ok": ok, "message": msg})

    @app.route("/api/blocked_users", methods=["DELETE"])
    @require_auth
    def delete_blocked_user():
        data = request.get_json(force=True)
        url = data.get("url", "").strip()
        if not url:
            return jsonify({"ok": False, "message": "URL ריק"}), 400
        ok, msg = remove_blocked_user(url)
        return jsonify({"ok": ok, "message": msg})

    return app


if __name__ == "__main__":
    init_db()
    app = create_app()
    log.info(f"פאנל הגדרות רץ על פורט {PANEL_PORT}")
    app.run(host="0.0.0.0", port=PANEL_PORT, debug=False)
