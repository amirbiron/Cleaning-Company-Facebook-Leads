import atexit
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from zoneinfo import ZoneInfo

# נתיב אבסולוטי יחסית למיקום הקובץ — עובד גם ב-Docker וגם מקומית,
# ולא תלוי ב-CWD שיכול להשתנות בזמן ריצה (Flask, threads).
_BASE_DIR = Path(__file__).resolve().parent
DB_PATH = _BASE_DIR / "data" / "leads.db"


def _now() -> datetime:
    """מחזיר את הזמן הנוכחי באזור הזמן המוגדר (TIMEZONE env var)."""
    tz_name = os.environ.get("TIMEZONE", "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")
    return datetime.now(tz=tz)


# ── Connection Pool (thread-local) ────────────────────────────
_local = threading.local()
_all_connections: list[sqlite3.Connection] = []
_conn_lock = threading.Lock()

def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH, timeout=10)
        _local.conn.execute("PRAGMA journal_mode=WAL")
        with _conn_lock:
            _all_connections.append(_local.conn)
    return _local.conn


def _close_all_connections():
    """סוגר את כל ה-connections הפתוחים — נקרא ב-atexit."""
    with _conn_lock:
        for conn in _all_connections:
            try:
                conn.close()
            except Exception:
                pass
        _all_connections.clear()


atexit.register(_close_all_connections)

# ── Init ──────────────────────────────────────────────────────
def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_posts (
                post_id TEXT PRIMARY KEY,
                group_name TEXT,
                seen_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sent_leads (
                post_id TEXT PRIMARY KEY,
                group_name TEXT,
                content TEXT,
                reason TEXT,
                sent_at TEXT,
                content_hash TEXT DEFAULT ''
            )
        """)
        # מיגרציה — הוספת עמודת content_hash ל-DB קיים
        try:
            conn.execute("ALTER TABLE sent_leads ADD COLUMN content_hash TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # העמודה כבר קיימת
        # אינדקס על content_hash — מאיץ חיפוש כפילויות לפי תוכן
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sent_leads_content_hash ON sent_leads(content_hash)")
        # אינדקס על seen_posts.seen_at — מאיץ cleanup של פוסטים ישנים
        conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_posts_seen_at ON seen_posts(seen_at)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('pre_filter', 'block')),
                created_at TEXT NOT NULL,
                UNIQUE(word, type)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                estimated_cost_usd REAL NOT NULL,
                call_type TEXT NOT NULL DEFAULT 'single',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS blocked_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                profile_url TEXT NOT NULL DEFAULT '' UNIQUE,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS group_health (
                group_url TEXT PRIMARY KEY,
                last_posts_count INTEGER DEFAULT 0,
                consecutive_empty INTEGER DEFAULT 0,
                last_lead_at TEXT,
                updated_at TEXT
            )
        """)
        # מיגרציה — הוספת עמודת profile_url ל-DBs קיימים
        cols = [r[1] for r in conn.execute("PRAGMA table_info(blocked_users)").fetchall()]
        if "profile_url" not in cols:
            # ערכים ישנים (שמות טקסטואליים) לא יתאימו ל-URL — שומרים אותם לפני מחיקה
            old_users = conn.execute("SELECT name FROM blocked_users").fetchall()
            old_names = [r[0] for r in old_users]
            # SQLite לא תומך ב-DROP CONSTRAINT, לכן יוצרים טבלה חדשה בלי UNIQUE על name
            conn.execute("""
                CREATE TABLE blocked_users_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    profile_url TEXT NOT NULL DEFAULT '' UNIQUE,
                    created_at TEXT NOT NULL
                )
            """)
            # לא מעבירים ערכים ישנים — שמות טקסטואליים לא יתאימו ל-URL פרופיל
            conn.execute("DROP TABLE blocked_users")
            conn.execute("ALTER TABLE blocked_users_new RENAME TO blocked_users")
            if old_names:
                import sys
                print(
                    f"[MIGRATION] blocked_users: {len(old_names)} רשומות ישנות (לפי שם) נמחקו. "
                    f"חסימה עובדת עכשיו לפי URL פרופיל. יש להוסיף מחדש דרך הפאנל/טלגרם: "
                    f"{', '.join(old_names)}",
                    file=sys.stderr,
                )

# ── CRUD ──────────────────────────────────────────────────────
def is_seen(post_id: str) -> bool:
    row = _get_conn().execute(
        "SELECT 1 FROM seen_posts WHERE post_id = ?", (post_id,)
    ).fetchone()
    return row is not None

def mark_seen(post_id: str, group_name: str):
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO seen_posts (post_id, group_name, seen_at) VALUES (?, ?, ?)",
            (post_id, group_name, _now().isoformat())
        )

def save_lead(post_id: str, group_name: str, content: str, reason: str,
              content_hash: str = ""):
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sent_leads (post_id, group_name, content, reason, sent_at, content_hash) VALUES (?, ?, ?, ?, ?, ?)",
            (post_id, group_name, content, reason, _now().isoformat(), content_hash)
        )

def is_lead_sent(post_id: str) -> bool:
    row = _get_conn().execute(
        "SELECT 1 FROM sent_leads WHERE post_id = ?", (post_id,)
    ).fetchone()
    return row is not None

def is_content_hash_sent(content_hash: str) -> bool:
    """בודק אם ליד עם אותו content_hash כבר נשלח — מונע כפילויות כשה-post_id משתנה בין סשנים."""
    if not content_hash:
        return False
    row = _get_conn().execute(
        "SELECT 1 FROM sent_leads WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return row is not None

def get_stats() -> dict:
    conn = _get_conn()
    seen = conn.execute("SELECT COUNT(*) FROM seen_posts").fetchone()[0]
    sent = conn.execute("SELECT COUNT(*) FROM sent_leads").fetchone()[0]
    return {"seen": seen, "sent": sent}


def get_daily_stats(today_prefix: str) -> dict:
    """מחזיר סטטיסטיקות של היום בלבד.
    today_prefix — מחרוזת בפורמט 'YYYY-MM-DD' לסינון לפי seen_at/sent_at.
    משתמש ב-BETWEEN במקום LIKE לביצועים טובים יותר ככל שהטבלה גדלה.
    """
    conn = _get_conn()
    next_day = (datetime.fromisoformat(today_prefix) + timedelta(days=1)).strftime('%Y-%m-%d')
    seen = conn.execute(
        "SELECT COUNT(*) FROM seen_posts WHERE seen_at >= ? AND seen_at < ?",
        (today_prefix, next_day),
    ).fetchone()[0]
    sent = conn.execute(
        "SELECT COUNT(*) FROM sent_leads WHERE sent_at >= ? AND sent_at < ?",
        (today_prefix, next_day),
    ).fetchone()[0]
    return {"seen": seen, "sent": sent}


# ── Groups CRUD ───────────────────────────────────────────────

def _normalize_group_url(raw_url: str) -> str:
    """ממיר URL לפורמט m.facebook.com ומנקה."""
    url = raw_url.strip().rstrip("/")
    # הוספת סכמה לפני החלפת דומיין, כדי למנוע כפילות prefix
    if not url.startswith("http"):
        if "facebook.com" in url:
            url = "https://" + url
        else:
            url = "https://m.facebook.com/groups/" + url
    # הסרת query params ו-fragment — לינקי שיתוף מכילים ?ref=share&rdid=...&#
    # חייבים לנקות לפני החלפת דומיין כדי לא לפגוע ב-URL מקודד בתוך share_url param
    parsed = urlparse(url)
    url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    url = url.rstrip("/")
    url = url.replace("www.facebook.com", "m.facebook.com")
    # טיפול בדומיין חשוף (facebook.com ללא subdomain)
    url = url.replace("://facebook.com", "://m.facebook.com")
    return url

def _extract_group_name(url: str) -> str:
    """מחלץ שם קבוצה מ-URL."""
    match = re.search(r"/groups/([^/?]+)", url)
    return match.group(1) if match else url

def _get_max_groups() -> int:
    """מחזיר את מגבלת הקבוצות. 0 = ללא הגבלה."""
    try:
        return int(os.environ.get("MAX_GROUPS", "0"))
    except (ValueError, TypeError):
        return 0


def count_groups() -> int:
    """מחזיר את מספר הקבוצות הפעילות ב-DB."""
    row = _get_conn().execute("SELECT COUNT(*) FROM groups").fetchone()
    return row[0] if row else 0


def add_group(raw_url: str) -> tuple[bool, str]:
    """מוסיף קבוצה ל-DB. מחזיר (success, message)."""
    url = _normalize_group_url(raw_url)
    name = _extract_group_name(url)
    max_groups = _get_max_groups()
    try:
        with _get_conn() as conn:
            # בדיקת כפילות לפני בדיקת מגבלה — כדי לא להציג "שדרג" על קבוצה שכבר קיימת
            if conn.execute("SELECT 1 FROM groups WHERE url = ?", (url,)).fetchone():
                return False, f"הקבוצה כבר קיימת: {url}"
            # בדיקת מגבלה + INSERT באותה טרנזקציה — מונע race condition
            if max_groups > 0:
                count = conn.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
                if count >= max_groups:
                    return False, f"הגעת למגבלת הקבוצות ({max_groups}). לשדרוג — פנה למנהל."
            conn.execute(
                "INSERT INTO groups (name, url, created_at) VALUES (?, ?, ?)",
                (name, url, _now().isoformat()),
            )
        _mark_feature_configured("groups")
        # אם השם הוא רק ספרות (מזהה URL) — מודיעים שיתעדכן בסריקה
        if name.isdigit():
            return True, f"קבוצה נוספה: {url}\nשם הקבוצה יעודכן אוטומטית בסריקה הבאה."
        return True, f"קבוצה נוספה: {name}\n{url}"
    except sqlite3.IntegrityError:
        return False, f"הקבוצה כבר קיימת: {url}"

def remove_group(raw_url: str) -> tuple[bool, str]:
    """מסיר קבוצה מה-DB. מחזיר (success, message)."""
    url = _normalize_group_url(raw_url)
    with _get_conn() as conn:
        deleted = conn.execute("DELETE FROM groups WHERE url = ?", (url,)).rowcount
    if deleted:
        return True, f"קבוצה הוסרה: {url}"
    return False, f"קבוצה לא נמצאה: {url}"

def _feature_was_configured(key: str) -> bool:
    """בודק אם פיצ'ר מסוים הופעל אי-פעם (דרך טבלת _config)."""
    row = _get_conn().execute(
        "SELECT 1 FROM _config WHERE key = ?", (key,)
    ).fetchone()
    return row is not None

def _mark_feature_configured(key: str):
    """מסמן פיצ'ר כמוגדר (נקרא בפעם הראשונה שמוסיפים שורה)."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO _config (key, value) VALUES (?, '1')", (key,)
        )

def update_group_name(url: str, name: str) -> bool:
    """מעדכן את שם הקבוצה ב-DB לפי URL. מחזיר True אם עודכן."""
    url = _normalize_group_url(url)
    with _get_conn() as conn:
        updated = conn.execute(
            "UPDATE groups SET name = ? WHERE url = ?", (name, url)
        ).rowcount
    return updated > 0

def get_db_groups() -> list[dict] | None:
    """מחזיר קבוצות מה-DB, או None אם מעולם לא הוגדרה קבוצה."""
    if not _feature_was_configured("groups"):
        return None
    rows = _get_conn().execute("SELECT name, url FROM groups ORDER BY id").fetchall()
    return [{"name": r[0], "url": r[1]} for r in rows]

# ── Keywords CRUD ─────────────────────────────────────────────

def ensure_keywords_migrated(kw_type: str, defaults: list[str]):
    """אם סוג מילות מפתח מעולם לא הוגדר, מעביר את ברירות המחדל ל-DB.
    חובה לקרוא לפני add_keyword הראשון — אחרת ברירות המחדל נעלמות.
    """
    if _feature_was_configured(f"keywords_{kw_type}"):
        return  # כבר הוגדר — אין צורך במיגרציה
    if not defaults:
        return
    now = _now().isoformat()
    with _get_conn() as conn:
        for word in defaults:
            word = word.strip().lower()
            if word:
                conn.execute(
                    "INSERT OR IGNORE INTO keywords (word, type, created_at) VALUES (?, ?, ?)",
                    (word, kw_type, now),
                )
    _mark_feature_configured(f"keywords_{kw_type}")

def add_keyword(word: str, kw_type: str) -> tuple[bool, str]:
    """מוסיף מילת מפתח ל-DB. kw_type: 'pre_filter' או 'block'."""
    word = word.strip().lower()
    if not word:
        return False, "מילה ריקה"
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO keywords (word, type, created_at) VALUES (?, ?, ?)",
                (word, kw_type, _now().isoformat()),
            )
        _mark_feature_configured(f"keywords_{kw_type}")
        label = "מילת מפתח" if kw_type == "pre_filter" else "מילה חסומה"
        return True, f"{label} נוספה: {word}"
    except sqlite3.IntegrityError:
        return False, f"המילה כבר קיימת: {word}"

def remove_keyword(word: str, kw_type: str) -> tuple[bool, str]:
    """מסיר מילת מפתח מה-DB."""
    word = word.strip().lower()
    with _get_conn() as conn:
        deleted = conn.execute(
            "DELETE FROM keywords WHERE word = ? AND type = ?", (word, kw_type)
        ).rowcount
    if deleted:
        return True, f"מילה הוסרה: {word}"
    return False, f"מילה לא נמצאה: {word}"

def get_db_keywords(kw_type: str) -> list[str] | None:
    """מחזיר מילות מפתח מה-DB לפי סוג, או None אם הסוג מעולם לא הוגדר."""
    if not _feature_was_configured(f"keywords_{kw_type}"):
        return None
    rows = _get_conn().execute(
        "SELECT word FROM keywords WHERE type = ? ORDER BY id", (kw_type,)
    ).fetchall()
    return [r[0] for r in rows]

# ── Blocked Users CRUD ────────────────────────────────────

def _normalize_profile_url(raw: str) -> str:
    """מנרמל URL פרופיל פייסבוק — מסיר פרמטרי tracking ומעביר ל-m.facebook.com.
    שומר על id= ב-profile.php URLs (משתמשים בלי vanity URL).
    """
    url = raw.strip()
    if not url:
        return ""
    # שמירת id= מ-profile.php לפני הסרת query params
    profile_id = ""
    if "profile.php" in url:
        match = re.search(r"[?&]id=(\d+)", url)
        if match:
            profile_id = match.group(1)
    # הסרת query params (fbclid, tracking וכו')
    url = url.split("?")[0].rstrip("/")
    # שחזור id= ל-profile.php
    if profile_id:
        url += f"?id={profile_id}"
    # נרמול דומיין
    if "facebook.com" in url and not url.startswith("http"):
        url = "https://" + url
    url = url.replace("www.facebook.com", "m.facebook.com")
    url = url.replace("://facebook.com", "://m.facebook.com")
    return url


def _extract_profile_name(url: str) -> str:
    """מחלץ שם פרופיל מ-URL (לתצוגה עד שהסקרייפר יעדכן)."""
    match = re.search(r"facebook\.com/(?:profile\.php\?id=)?([^/?]+)", url)
    return match.group(1) if match else url


def add_blocked_user(profile_url: str, display_name: str = "") -> tuple[bool, str]:
    """מוסיף מפרסם לרשימה שחורה לפי URL פרופיל. מחזיר (success, message)."""
    url = _normalize_profile_url(profile_url)
    if not url:
        return False, "URL ריק"
    name = display_name.strip() or _extract_profile_name(url)
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO blocked_users (name, profile_url, created_at) VALUES (?, ?, ?)",
                (name, url, _now().isoformat()),
            )
        return True, f"מפרסם נחסם: {name}"
    except sqlite3.IntegrityError:
        return False, f"המפרסם כבר חסום: {url}"


def remove_blocked_user(profile_url: str) -> tuple[bool, str]:
    """מסיר מפרסם מהרשימה השחורה לפי URL. מחזיר (success, message)."""
    url = _normalize_profile_url(profile_url)
    with _get_conn() as conn:
        deleted = conn.execute(
            "DELETE FROM blocked_users WHERE profile_url = ?", (url,)
        ).rowcount
    if deleted:
        return True, f"מפרסם הוסר מהחסימה: {url}"
    return False, f"מפרסם לא נמצא ברשימה: {url}"


def get_blocked_users() -> list[dict]:
    """מחזיר את כל המפרסמים החסומים כרשימת dicts עם name ו-profile_url."""
    rows = _get_conn().execute(
        "SELECT name, profile_url FROM blocked_users ORDER BY id"
    ).fetchall()
    return [{"name": r[0], "profile_url": r[1]} for r in rows]


# ── Config (key-value) ────────────────────────────────────────

def get_config(key: str, default=None):
    """מחזיר ערך מטבלת _config, או default אם המפתח לא קיים.
    ברירת מחדל ל-default: None (אפשר להעביר כל ערך, כולל sentinel).
    """
    row = _get_conn().execute(
        "SELECT value FROM _config WHERE key = ?", (key,)
    ).fetchone()
    if row is not None:
        return row[0]
    return default


def get_config_by_prefix(prefix: str) -> list[tuple[str, str]]:
    """מחזיר כל הרשומות מטבלת _config שה-key מתחיל ב-prefix.
    מחזיר רשימת tuples של (key, value).
    """
    try:
        rows = _get_conn().execute(
            "SELECT key, value FROM _config WHERE substr(key, 1, ?) = ?",
            (len(prefix), prefix),
        ).fetchall()
    except Exception:
        return []
    return rows

def set_config(key: str, value: str):
    """שומר ערך בטבלת _config (מחליף אם קיים)."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO _config (key, value) VALUES (?, ?)",
            (key, value),
        )

# ── Encrypted Config (credentials) ──────────────────────────

def _get_fernet():
    """מחזיר Fernet instance. המפתח נגזר מ-ENCRYPTION_KEY env var.
    אם המשתנה לא מוגדר — חוזר None (ללא הצפנה, backward-compatible).
    """
    key = os.environ.get("ENCRYPTION_KEY", "").strip()
    if not key:
        return None
    try:
        import base64
        import hashlib
        from cryptography.fernet import Fernet
        # גוזר מפתח 32 בתים מהסיסמה (SHA-256) ומקודד ל-base64 urlsafe
        derived = hashlib.sha256(key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(derived)
        return Fernet(fernet_key)
    except ImportError:
        return None
    except Exception:
        return None

def set_config_encrypted(key: str, value: str):
    """שומר ערך מוצפן בטבלת _config. אם אין ENCRYPTION_KEY — שומר רגיל."""
    f = _get_fernet()
    if f and value:
        encrypted = f.encrypt(value.encode()).decode()
        set_config(key, f"enc:{encrypted}")
    else:
        set_config(key, value)

def get_config_encrypted(key: str, default=None):
    """קורא ערך מוצפן מטבלת _config. אם הערך לא מוצפן — מחזיר כמו שהוא."""
    raw = get_config(key, default)
    if raw is None or raw is default:
        return raw
    if isinstance(raw, str) and raw.startswith("enc:"):
        f = _get_fernet()
        if f:
            try:
                return f.decrypt(raw[4:].encode()).decode()
            except Exception:
                return None  # מפתח הצפנה השתנה — None כדי שה-caller ייפול ל-env var
    return raw

# ── API Usage Tracking ────────────────────────────────────────

# עלויות לכל 1M טוקנים (USD) — לפי מחירון OpenAI
_MODEL_COSTS = {
    "gpt-4.1-mini":  {"input": 0.40, "output": 1.60},
    "gpt-4o-mini":   {"input": 0.15, "output": 0.60},
}
_DEFAULT_COST = {"input": 0.40, "output": 1.60}  # fallback למודל לא מוכר


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """מחשב עלות משוערת ב-USD לפי מודל וטוקנים."""
    costs = _MODEL_COSTS.get(model, _DEFAULT_COST)
    return (prompt_tokens * costs["input"] + completion_tokens * costs["output"]) / 1_000_000


def save_api_usage(model: str, prompt_tokens: int, completion_tokens: int,
                   total_tokens: int, call_type: str = "single"):
    """שומר רשומת שימוש ב-API."""
    cost = _estimate_cost(model, prompt_tokens, completion_tokens)
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO api_usage (model, prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd, call_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (model, prompt_tokens, completion_tokens, total_tokens, cost, call_type, _now().isoformat()),
        )


def get_usage_stats() -> dict:
    """מחזיר סטטיסטיקות שימוש מצטברות."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
        "COALESCE(SUM(total_tokens),0), COALESCE(SUM(estimated_cost_usd),0), COUNT(*) "
        "FROM api_usage"
    ).fetchone()
    return {
        "prompt_tokens": row[0],
        "completion_tokens": row[1],
        "total_tokens": row[2],
        "total_cost_usd": row[3],
        "total_calls": row[4],
    }


def get_daily_usage_stats(today_prefix: str) -> dict:
    """מחזיר סטטיסטיקות שימוש של היום בלבד."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
        "COALESCE(SUM(total_tokens),0), COALESCE(SUM(estimated_cost_usd),0), COUNT(*) "
        "FROM api_usage WHERE created_at LIKE ?",
        (f"{today_prefix}%",),
    ).fetchone()
    return {
        "prompt_tokens": row[0],
        "completion_tokens": row[1],
        "total_tokens": row[2],
        "total_cost_usd": row[3],
        "total_calls": row[4],
    }


def get_usage_by_model() -> list[dict]:
    """מחזיר פירוט שימוש לפי מודל."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT model, SUM(prompt_tokens), SUM(completion_tokens), "
        "SUM(total_tokens), SUM(estimated_cost_usd), COUNT(*) "
        "FROM api_usage GROUP BY model ORDER BY SUM(estimated_cost_usd) DESC"
    ).fetchall()
    return [
        {
            "model": r[0],
            "prompt_tokens": r[1],
            "completion_tokens": r[2],
            "total_tokens": r[3],
            "total_cost_usd": r[4],
            "total_calls": r[5],
        }
        for r in rows
    ]


# ── Group Health ──────────────────────────────────────────────

def update_group_health(group_url: str, posts_count: int):
    """מעדכן את מצב הבריאות של קבוצה אחרי סריקה.
    אם posts_count == 0, מגדיל את consecutive_empty ב-1.
    אחרת, מאפס את consecutive_empty ל-0.
    """
    now = _now().isoformat()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT consecutive_empty FROM group_health WHERE group_url = ?",
            (group_url,),
        ).fetchone()
        if row is None:
            consecutive = 1 if posts_count == 0 else 0
            conn.execute(
                "INSERT INTO group_health (group_url, last_posts_count, consecutive_empty, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (group_url, posts_count, consecutive, now),
            )
        else:
            consecutive = (row[0] + 1) if posts_count == 0 else 0
            conn.execute(
                "UPDATE group_health SET last_posts_count = ?, consecutive_empty = ?, updated_at = ? "
                "WHERE group_url = ?",
                (posts_count, consecutive, now, group_url),
            )


def update_group_last_lead(group_url: str):
    """מעדכן את last_lead_at של קבוצה — נקרא כשנשלח ליד מהקבוצה.
    אם אין רשומה לקבוצה — יוצר אותה (INSERT).
    """
    now = _now().isoformat()
    with _get_conn() as conn:
        affected = conn.execute(
            "UPDATE group_health SET last_lead_at = ? WHERE group_url = ?",
            (now, group_url),
        ).rowcount
        if affected == 0:
            conn.execute(
                "INSERT INTO group_health (group_url, last_posts_count, consecutive_empty, last_lead_at, updated_at) "
                "VALUES (?, 0, 0, ?, ?)",
                (group_url, now, now),
            )


def get_all_group_health() -> list[dict]:
    """מחזיר מצב בריאות של כל הקבוצות."""
    rows = _get_conn().execute(
        "SELECT group_url, last_posts_count, consecutive_empty, last_lead_at, updated_at "
        "FROM group_health ORDER BY consecutive_empty DESC"
    ).fetchall()
    return [
        {
            "group_url": r[0],
            "last_posts_count": r[1],
            "consecutive_empty": r[2],
            "last_lead_at": r[3],
            "updated_at": r[4],
        }
        for r in rows
    ]


# ── Cleanup ───────────────────────────────────────────────────
def cleanup_old_posts(days: int = 30) -> int:
    """מוחק רשומות ישנות מטבלת seen_posts (ברירת מחדל: מעל 30 יום).
    מריץ VACUUM כשנמחקו מעל 1000 רשומות — מקטין את גודל קובץ ה-DB."""
    cutoff = (_now() - timedelta(days=days)).isoformat()
    with _get_conn() as conn:
        deleted = conn.execute(
            "DELETE FROM seen_posts WHERE seen_at < ?", (cutoff,)
        ).rowcount
    if deleted > 1000:
        _get_conn().execute("VACUUM")
    return deleted
