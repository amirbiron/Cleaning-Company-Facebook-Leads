import asyncio
import os
import re
import signal
import sys
import threading
from datetime import datetime, time, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from zoneinfo import ZoneInfo
from database import (
    init_db, is_seen, is_lead_sent, is_content_hash_sent, mark_seen, save_lead,
    get_stats, get_daily_stats,
    cleanup_old_posts, add_group, remove_group, get_db_groups,
    add_keyword, remove_keyword, get_db_keywords, ensure_keywords_migrated,
    get_config, set_config, get_config_encrypted, get_config_by_prefix,
    get_usage_stats, get_daily_usage_stats, get_usage_by_model,
    add_blocked_user, remove_blocked_user, get_blocked_users,
    update_group_health, update_group_last_lead, get_all_group_health,
)
from logger import get_logger

log = get_logger("Main")

INTERVAL_MINUTES = int(os.environ.get("INTERVAL_MINUTES", "10"))
TIMEZONE_NAME = os.environ.get("TIMEZONE", "UTC")
QUIET_HOURS = os.environ.get("QUIET_HOURS", "").strip()
TELEGRAM_CONTROL = os.environ.get("TELEGRAM_CONTROL", "1").strip().lower() not in ("0", "false", "no", "off")

HEALTH_PORT = int(os.environ.get("HEALTH_PORT", os.environ.get("PORT", "8080")))
DEVELOPER_CHAT_ID = os.environ.get("DEVELOPER_CHAT_ID", "").strip()
# מגבלת זמן אוטומציה יומית (בדקות). 0 = ללא מגבלה.
# מודד זמן סריקה בפועל (דפדפן פתוח), לא כולל המתנה בין סבבים.
DAILY_AUTOMATION_LIMIT_MINUTES = int(os.environ.get("DAILY_AUTOMATION_LIMIT_MINUTES", "0"))
_panel_url_raw = os.environ.get("PANEL_URL", "").strip()
PANEL_URL = _panel_url_raw if _panel_url_raw.startswith(("http://", "https://")) else ""

# ── סטטוס סריקה בזמן אמת — נגיש מהפאנל דרך /api/scan-status ──
# מילון שמתעדכן לאורך כל שלבי הסריקה. Flask קורא אותו (thread-safe לקריאה).
scan_progress: dict = {
    "active": False,
    "phase": "idle",          # idle | scraping | filtering | classifying | done
    "phase_label": "ממתין",
    "current_group": "",
    "current_group_index": 0,
    "total_groups": 0,
    "groups_done": [],        # [{"name": ..., "posts_found": ...}, ...]
    "total_posts_found": 0,
    "new_posts": 0,
    "posts_to_classify": 0,
    "leads_sent": 0,
    "started_at": None,       # ISO string
    "finished_at": None,      # ISO string
    "error": "",
}

# כשמריצים python main.py, המודול נטען כ-__main__ ולא כ-"main".
# פאנל Flask עושה `from main import scan_progress` — שיוצר מודול "main" נפרד
# עם dict חדש שלעולם לא מתעדכן → הפאנל תמיד מראה "ממתין".
# הרישום כאן מבטיח ששני השמות מצביעים לאותו מודול.
sys.modules.setdefault("main", sys.modules[__name__])

FB_EMAIL = os.environ.get("FB_EMAIL")
FB_PASSWORD = os.environ.get("FB_PASSWORD")

# מילים חוסמות — פוסט שמכיל מילה מהרשימה לא יישלח ל-AI
# ניתן לערוך דרך משתנה סביבה BLOCK_KEYWORDS (מופרד בפסיקים)
# או דינמית דרך פקודות טלגרם /block, /unblock
_block_env = os.environ.get("BLOCK_KEYWORDS", "").strip()
_BLOCK_KEYWORDS_ENV = [w.strip().lower() for w in _block_env.split(",") if w.strip()] if _block_env else []

# מילות מפתח לסינון מוקדם — רק פוסטים שמכילים לפחות מילה אחת יעברו ל-AI
# ניתן להגדיר דרך משתנה סביבה PRE_FILTER_KEYWORDS (מופרד בפסיקים),
# דרך פאנל ההגדרות, או דרך פקודות טלגרם /add_keyword, /remove_keyword.
# אם לא הוגדר — רשימה ריקה (= אין סינון מוקדם, הכל עובר ל-AI).
_pre_filter_env = os.environ.get("PRE_FILTER_KEYWORDS", "").strip()
_PRE_FILTER_KEYWORDS_DEFAULT = (
    [w.strip().lower() for w in _pre_filter_env.split(",") if w.strip()]
    if _pre_filter_env else []
)

def _load_force_send_keywords() -> list[str]:
    """טוען מילות 'שלח תמיד' מ-DB. פוסט שמכיל מילה כזו נשלח ישירות ללא AI."""
    import json as _json
    try:
        raw = get_config("force_send_keywords")
        if raw:
            return _json.loads(raw)
    except Exception:
        pass
    return []

# ── מילים חמות (hot words) — ליד חם עם התראה ────────────────

def _load_hot_words() -> list[str]:
    """טוען מילים חמות מ-DB. פוסט שמכיל מילה כזו מסומן כ'ליד חם'."""
    import json as _json
    try:
        raw = get_config("hot_words")
        if raw:
            return _json.loads(raw)
    except Exception:
        pass
    return []

def _save_hot_words(words: list[str]):
    """שומר מילים חמות ב-DB."""
    import json as _json
    from database import set_config
    set_config("hot_words", _json.dumps(words))

def add_hot_word(word: str) -> tuple[bool, str]:
    """מוסיף מילה חמה. מחזיר (success, message)."""
    word = word.strip().lower()
    if not word:
        return False, "מילה ריקה"
    words = _load_hot_words()
    if word in words:
        return False, f"המילה כבר קיימת: {word}"
    words.append(word)
    _save_hot_words(words)
    return True, f"מילה חמה נוספה: {word}"

def remove_hot_word(word: str) -> tuple[bool, str]:
    """מסיר מילה חמה. מחזיר (success, message)."""
    word = word.strip().lower()
    words = _load_hot_words()
    if word not in words:
        return False, f"מילה לא נמצאה: {word}"
    words.remove(word)
    _save_hot_words(words)
    return True, f"מילה חמה הוסרה: {word}"

def matches_hot_word(text: str) -> str | None:
    """בודק אם הפוסט מכיל מילה חמה.
    מחזיר את המילה שנמצאה או None.
    """
    with _kw_lock:
        hot_words = _keywords_state.get("hot_words", [])
    if not hot_words:
        return None
    text_lower = text.lower()
    for word in hot_words:
        if word in text_lower:
            return word
    return None

def _save_force_send_keywords(keywords: list[str]):
    """שומר מילות 'שלח תמיד' ב-DB."""
    import json as _json
    from database import set_config
    set_config("force_send_keywords", _json.dumps(keywords))

def add_force_send_keyword(word: str) -> tuple[bool, str]:
    """מוסיף מילת 'שלח תמיד'. מחזיר (success, message)."""
    word = word.strip().lower()
    if not word:
        return False, "מילה ריקה"
    keywords = _load_force_send_keywords()
    if word in keywords:
        return False, f"המילה כבר קיימת: {word}"
    keywords.append(word)
    _save_force_send_keywords(keywords)
    return True, f"מילת 'שלח תמיד' נוספה: {word}"

def remove_force_send_keyword(word: str) -> tuple[bool, str]:
    """מסיר מילת 'שלח תמיד'. מחזיר (success, message)."""
    word = word.strip().lower()
    keywords = _load_force_send_keywords()
    if word not in keywords:
        return False, f"מילה לא נמצאה: {word}"
    keywords.remove(word)
    _save_force_send_keywords(keywords)
    return True, f"מילה הוסרה: {word}"

# ── מילות "שלח תמיד" לקבוצה ספציפית ──────────────────────────
# נשמרות ב-_config עם מפתח "force_send_group:<url>" — כל קבוצה מקבלת רשימה משלה.

def _force_send_group_key(group_url: str) -> str:
    """מחזיר מפתח _config עבור מילות force_send של קבוצה."""
    from database import _normalize_group_url
    return f"force_send_group:{_normalize_group_url(group_url)}"

def _load_group_force_send_keywords(group_url: str) -> list[str]:
    """טוען מילות 'שלח תמיד' ספציפיות לקבוצה."""
    import json as _json
    try:
        raw = get_config(_force_send_group_key(group_url))
        if raw:
            return _json.loads(raw)
    except Exception:
        pass
    return []

def _save_group_force_send_keywords(group_url: str, keywords: list[str]):
    """שומר מילות 'שלח תמיד' ספציפיות לקבוצה."""
    import json as _json
    set_config(_force_send_group_key(group_url), _json.dumps(keywords))

def add_group_force_send_keyword(group_url: str, word: str) -> tuple[bool, str]:
    """מוסיף מילת 'שלח תמיד' לקבוצה ספציפית."""
    word = word.strip().lower()
    if not word:
        return False, "מילה ריקה"
    keywords = _load_group_force_send_keywords(group_url)
    if word in keywords:
        return False, f"המילה כבר קיימת: {word}"
    keywords.append(word)
    _save_group_force_send_keywords(group_url, keywords)
    return True, f"מילת 'שלח תמיד' נוספה לקבוצה: {word}"

def remove_group_force_send_keyword(group_url: str, word: str) -> tuple[bool, str]:
    """מסיר מילת 'שלח תמיד' מקבוצה ספציפית."""
    word = word.strip().lower()
    keywords = _load_group_force_send_keywords(group_url)
    if word not in keywords:
        return False, f"מילה לא נמצאה: {word}"
    keywords.remove(word)
    _save_group_force_send_keywords(group_url, keywords)
    return True, f"מילה הוסרה מהקבוצה: {word}"

def get_all_group_force_send() -> dict[str, list[str]]:
    """מחזיר מילון של כל מילות force_send לפי קבוצה.
    {url: [keyword, ...], ...} — רק קבוצות שיש להן מילים.
    """
    import json as _json
    prefix = "force_send_group:"
    rows = get_config_by_prefix(prefix)
    result = {}
    for key, value in rows:
        url = key[len(prefix):]
        try:
            kws = _json.loads(value)
            if kws:
                result[url] = kws
        except Exception:
            pass
    return result

def matches_force_send(text: str, group_url: str = "") -> str | None:
    """בודק אם הפוסט מכיל מילת 'שלח תמיד'.
    בודק קודם מילות גלובליות, אח"כ מילות קבוצה ספציפית.
    מחזיר את המילה שנמצאה או None.
    """
    with _kw_lock:
        force_kws = _keywords_state["force_send"]
        group_force_kws = _keywords_state["group_force_send"]
    text_lower = text.lower()
    # בדיקת מילות גלובליות
    if force_kws:
        for kw in force_kws:
            if kw in text_lower:
                return kw
    # בדיקת מילות קבוצה ספציפית
    if group_url and group_force_kws:
        from database import _normalize_group_url
        normalized = _normalize_group_url(group_url)
        group_kws = group_force_kws.get(normalized, [])
        for kw in group_kws:
            if kw in text_lower:
                return kw
    return None


def _load_block_keywords() -> list[str]:
    """טוען מילים חוסמות מ-DB עם fallback למשתנה סביבה.
    None מ-DB = מעולם לא הוגדר → fallback.  [] = המשתמש רוקן הכל → רשימה ריקה.
    """
    try:
        db_kw = get_db_keywords("block")
        if db_kw is not None:
            return db_kw
    except Exception:
        pass
    return _BLOCK_KEYWORDS_ENV

def _load_pre_filter_keywords() -> list[str]:
    """טוען מילות מפתח לסינון מוקדם מ-DB עם fallback לרשימת ברירת מחדל.
    None מ-DB = מעולם לא הוגדר → fallback.  [] = המשתמש רוקן הכל → רשימה ריקה.
    """
    try:
        db_kw = get_db_keywords("pre_filter")
        if db_kw is not None:
            return db_kw
    except Exception:
        pass
    return _PRE_FILTER_KEYWORDS_DEFAULT

# טעינה ראשונית — יתעדכנו דינמית בכל סבב.
# כל הגישה דרך _kw_lock כדי למנוע race condition בין Flask thread ל-main loop.
def _load_blocked_users() -> list[dict]:
    """טוען מפרסמים חסומים מ-DB. רשימה ריקה אם ה-DB לא זמין."""
    try:
        return get_blocked_users()
    except Exception:
        return []

_kw_lock = threading.Lock()
_keywords_state: dict = {
    "block": _load_block_keywords(),
    "pre_filter": _load_pre_filter_keywords(),
    "force_send": _load_force_send_keywords(),
    "group_force_send": get_all_group_force_send(),
    "blocked_users": _load_blocked_users(),
    "hot_words": _load_hot_words(),
}
# aliases — backward-compatible read-only access (שומרים את השמות הישנים לקריאה בלבד)
BLOCK_KEYWORDS = _keywords_state["block"]
PRE_FILTER_KEYWORDS = _keywords_state["pre_filter"]
FORCE_SEND_KEYWORDS = _keywords_state["force_send"]
GROUP_FORCE_SEND_KEYWORDS: dict[str, list[str]] = _keywords_state["group_force_send"]

def reload_keywords():
    """טוען מחדש את מילות המפתח מ-DB (נקרא אחרי שינוי דרך טלגרם).
    עדכון אטומי — כל הרשימות מתחלפות תחת lock אחד.
    """
    global BLOCK_KEYWORDS, PRE_FILTER_KEYWORDS, FORCE_SEND_KEYWORDS, GROUP_FORCE_SEND_KEYWORDS
    new_block = _load_block_keywords()
    new_pre = _load_pre_filter_keywords()
    new_force = _load_force_send_keywords()
    new_group_force = get_all_group_force_send()
    new_blocked_users = _load_blocked_users()
    new_hot_words = _load_hot_words()
    with _kw_lock:
        _keywords_state["block"] = new_block
        _keywords_state["pre_filter"] = new_pre
        _keywords_state["force_send"] = new_force
        _keywords_state["group_force_send"] = new_group_force
        _keywords_state["blocked_users"] = new_blocked_users
        _keywords_state["hot_words"] = new_hot_words
        BLOCK_KEYWORDS = new_block
        PRE_FILTER_KEYWORDS = new_pre
        FORCE_SEND_KEYWORDS = new_force
        GROUP_FORCE_SEND_KEYWORDS = new_group_force

def passes_keyword_filter(text: str) -> bool:
    """בודק אם הפוסט מכיל לפחות מילת מפתח אחת.
    אם הרשימה ריקה — מעביר הכל (אין סינון מוקדם).
    """
    with _kw_lock:
        kws = _keywords_state["pre_filter"]
    if not kws:
        return True
    text_lower = text.lower()
    return any(kw in text_lower for kw in kws)

def is_blocked(text: str) -> bool:
    """בודק אם הפוסט מכיל מילה חסומה — אם כן, לא יישלח ל-AI."""
    with _kw_lock:
        kws = _keywords_state["block"]
    if not kws:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in kws)


def is_user_blocked(author_url: str) -> bool:
    """בודק אם המפרסם נמצא ברשימה השחורה לפי URL פרופיל.
    רשימה ריקה = אין חסימה.
    """
    if not author_url:
        return False
    with _kw_lock:
        users = _keywords_state["blocked_users"]
    if not users:
        return False
    # נרמול ה-URL לפני השוואה (הסרת query params, www→m)
    from database import _normalize_profile_url
    normalized = _normalize_profile_url(author_url)
    blocked_urls = {u["profile_url"] for u in users}
    return normalized in blocked_urls


# ── סינון לפי גיל פוסט ──────────────────────────────────────

# ביטויים לחילוץ גיל פוסט מתוך הטקסט (עברית + אנגלית)
_AGE_HEBREW_RE = re.compile(
    # חשוב: בלי prefixים אמביוולנטיים (למשל "שנ" = "שניות" וגם "שנים").
    # לכן היחידות כאן מפורשות/חד-משמעיות, והחילוץ מתבצע על "שורה עצמאית" בלבד.
    r'^לפני\s+(?:כ[-־]?\s*)?(\d+)\s*'
    r'(שני(?:ה|ות)|דק(?:ה|ות)|שע(?:ה|ות)|י(?:ום|מים)|שבוע(?:ות)?|חודש(?:ים)?|שנ(?:ה|ים))'
    r'\b(?:\s*[·•].*)?$',
    re.IGNORECASE
)
_AGE_HEBREW_SINGLE_RE = re.compile(
    r'^לפני\s+(שנייה|שניות|דקה|שעה|יום|יומיים|שבוע|חודש|שנה|שנתיים)\b(?:\s*[·•].*)?$',
    re.IGNORECASE
)
# פורמט קצר — שורה עצמאית בלבד: "3h", "2d" וכו'
_AGE_SHORT_RE = re.compile(r'^(\d+)\s*([hdmw])$', re.IGNORECASE)
# מעוגן לשורה עצמאית — כדי לא לתפוס "30 day trial" או "3 month project" מתוך גוף הפוסט
_AGE_ENGLISH_RE = re.compile(
    r'^(\d+)\s*(min|mins|minute|minutes|hr|hrs|hour|hours|day|days|wk|wks|week|weeks|month|months)\b'
    r'(?:\s+ago)?(?:\s*[·•].*)?$',
    re.IGNORECASE,
)


def extract_post_age_days(text: str) -> float | None:
    """מחלץ גיל פוסט (בימים) מתוך הטקסט. מחזיר None אם לא נמצא timestamp.

    פייסבוק מובייל מציג timestamps כמו "לפני 3 שעות", "לפני 2 ימים", "3h", "2d".
    הפונקציה סורקת את התוכן ומחזירה את הגיל בימים (float).
    """
    if not text:
        return None

    # חשוב: כדי להימנע מ-false positives מתוך גוף הפוסט, מחפשים רק בשורות עצמאיות.
    # (בדומה ל-_TIMESTAMP_RE ב-scraper.py שמוגדר כ-line-anchored)
    for line in text.split('\n'):
        s = line.strip()
        if not s:
            continue

        # מילים עצמאיות
        if s in ("עכשיו", "Just now"):
            return 0.0
        if s in ("אתמול", "Yesterday"):
            return 1.0

        # פורמט קצר — "3h", "2d" וכו' (שורה עצמאית)
        m = _AGE_SHORT_RE.fullmatch(s)
        if m:
            num = int(m.group(1))
            unit = m.group(2).lower()
            if unit == 'm':    # minutes
                return num / 1440.0
            if unit == 'h':    # hours
                return num / 24.0
            if unit == 'd':    # days
                return float(num)
            if unit == 'w':    # weeks
                return num * 7.0

        # אנגלית — "3 hours", "2 days" וכו' (שורה עצמאית)
        m = _AGE_ENGLISH_RE.fullmatch(s)
        if m:
            num = int(m.group(1))
            unit = m.group(2).lower()
            if unit in ("min", "mins", "minute", "minutes"):
                return num / 1440.0
            if unit in ("hr", "hrs", "hour", "hours"):
                return num / 24.0
            if unit in ("day", "days"):
                return float(num)
            if unit in ("wk", "wks", "week", "weeks"):
                return num * 7.0
            if unit in ("month", "months"):
                return num * 30.0

        # עברית — "לפני X [יחידה]" (שורה עצמאית)
        m = _AGE_HEBREW_RE.fullmatch(s)
        if m:
            num = int(m.group(1))
            unit = m.group(2)
            if unit.startswith("שני"):       # שניות
                return num / 86400.0
            if unit.startswith("דק"):        # דקות
                return num / 1440.0
            if unit.startswith("שע"):        # שעות
                return num / 24.0
            if unit.startswith("י"):         # יום/ימים
                return float(num)
            if unit.startswith("שבוע"):      # שבוע/שבועות
                return num * 7.0
            if unit.startswith("חודש"):      # חודש/חודשים
                return num * 30.0
            if unit.startswith("שנ"):        # שנה/שנים
                return num * 365.0

        # עברית — יחידות בודדות ללא מספר ("לפני דקה", "לפני שעה", "לפני יומיים")
        m = _AGE_HEBREW_SINGLE_RE.fullmatch(s)
        if m:
            word = m.group(1)
            if word in ("שנייה", "שניות"):
                return 1 / 86400.0
            if word == "דקה":
                return 1 / 1440.0
            if word == "שעה":
                return 1 / 24.0
            if word == "יום":
                return 1.0
            if word == "יומיים":
                return 2.0
            if word == "שבוע":
                return 7.0
            if word == "חודש":
                return 30.0
            if word == "שנה":
                return 365.0
            if word == "שנתיים":
                return 730.0

    return None


def _load_max_post_age() -> int | None:
    """טוען הגדרת גיל מקסימלי לפוסט (בימים) מ-DB. מחזיר None אם לא מוגדר."""
    val = get_config("max_post_age_days")
    if val is not None:
        try:
            days = int(val)
            return days if days > 0 else None
        except (ValueError, TypeError):
            pass
    return None


def is_post_too_old(text: str, max_days: int) -> bool:
    """בודק אם פוסט ישן מדי לפי גיל מקסימלי.
    מחזיר True אם הפוסט ישן מדי (= לא לשלוח).
    אם לא ניתן לקבוע את הגיל — מעביר את הפוסט (שמרני).
    """
    age = extract_post_age_days(text)
    if age is None:
        return False  # לא יודעים את הגיל → לא מסננים
    return age >= max_days


def _now_local() -> datetime:
    try:
        tz = ZoneInfo(TIMEZONE_NAME)
    except Exception:
        tz = ZoneInfo("UTC")
    return datetime.now(tz=tz)

def _parse_hhmm(value: str) -> time:
    value = value.strip()
    if not value:
        raise ValueError("empty time")

    if ":" in value:
        hh, mm = value.split(":", 1)
    else:
        hh, mm = value, "0"

    h = int(hh)
    m = int(mm)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"invalid time: {value}")
    return time(hour=h, minute=m)

def _parse_quiet_hours(spec: str) -> tuple[time, time] | None:
    spec = (spec or "").strip()
    if not spec:
        return None

    # Accept common formats: "02:00-07:00", "2-7", "02-07", "2:00-7:00"
    if "-" not in spec:
        raise ValueError("QUIET_HOURS must look like '02:00-07:00'")

    start_s, end_s = spec.split("-", 1)
    start_t = _parse_hhmm(start_s)
    end_t = _parse_hhmm(end_s)
    return start_t, end_t

def _is_quiet_now(now: datetime, quiet: tuple[time, time]) -> bool:
    start_t, end_t = quiet
    now_t = now.timetz().replace(tzinfo=None)

    if start_t == end_t:
        # Treat as "always quiet" (config means full-day).
        return True

    if start_t < end_t:
        return start_t <= now_t < end_t

    # Wraps over midnight (e.g., 22:00-06:00)
    return now_t >= start_t or now_t < end_t

def _seconds_until_quiet_end(now: datetime, quiet: tuple[time, time]) -> float:
    start_t, end_t = quiet

    today_end = now.replace(hour=end_t.hour, minute=end_t.minute, second=0, microsecond=0)

    if start_t < end_t:
        # Same-day window.
        end_dt = today_end
    else:
        # Wraps midnight. If we're after start, end is next day; otherwise it's today.
        now_t = now.timetz().replace(tzinfo=None)
        end_dt = today_end if now_t < end_t else (today_end + timedelta(days=1))

    return max(0.0, (end_dt - now).total_seconds())

def _parse_allowed_chat_ids(spec: str | None) -> set[int]:
    """Parses TELEGRAM_CHAT_ID (supports '123', '-100...', '123,456')."""
    if not spec:
        return set()
    parts = []
    for chunk in spec.replace("\n", ",").replace(" ", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    ids: set[int] = set()
    for p in parts:
        try:
            ids.add(int(p))
        except Exception:
            continue
    return ids

# ── כפתורי תפריט (Inline Keyboard) ──────────────────────────
def _main_menu_buttons() -> list[list[dict]]:
    """מחזיר כפתורי תפריט ראשי."""
    return [
        [
            {"text": "\U0001f50d סריקה עכשיו", "callback_data": "scan"},
            {"text": "\U0001f4ca סטטוס", "callback_data": "status"},
        ],
        [
            {"text": "\U0001f4cb דו\"ח יומי", "callback_data": "daily_report"},
            {"text": "\u2699\ufe0f הגדרות", "callback_data": "settings"},
        ],
        *(
            [[{"text": "\U0001f310 פאנל", "url": PANEL_URL}]]
            if PANEL_URL else []
        ),
    ]


def _settings_menu_buttons() -> list[list[dict]]:
    """מחזיר כפתורי תפריט הגדרות."""
    return [
        [
            {"text": "\U0001f4cb קבוצות", "callback_data": "groups"},
            {"text": "\U0001f50d מילות מפתח", "callback_data": "keywords"},
        ],
        [
            {"text": "\U0001f6ab מילים חוסמות", "callback_data": "blocked"},
            {"text": "\U0001f6d1 מפרסמים חסומים", "callback_data": "blocked_users"},
        ],
        [
            {"text": "\U0001f519 חזרה", "callback_data": "menu"},
        ],
    ]


def _back_to_menu_button() -> list[list[dict]]:
    """כפתור חזרה לתפריט ראשי."""
    return [[{"text": "\U0001f519 חזרה לתפריט", "callback_data": "menu"}]]


def _build_status_text(shared_state: dict) -> str:
    """בונה טקסט סטטוס — משותף לפקודה ולכפתור."""
    stats = get_stats()
    now = _now_local()
    quiet = shared_state.get("quiet")
    quiet_now = bool(quiet and _is_quiet_now(now, quiet))
    qh_db = get_config("quiet_hours", _CONFIG_NOT_SET)
    quiet_spec = qh_db if qh_db is not _CONFIG_NOT_SET else QUIET_HOURS
    quiet_desc = f"{quiet_spec} ({TIMEZONE_NAME})" if quiet_spec else "(disabled)"
    last_started = shared_state.get("last_scan_started")
    last_finished = shared_state.get("last_scan_finished")
    in_progress = bool(shared_state.get("scan_in_progress"))
    with _kw_lock:
        _bl = _keywords_state["block"]
        _pf = _keywords_state["pre_filter"]
    block_desc = ", ".join(_bl) if _bl else "(ריק)"
    from scraper import GROUPS
    groups_count = len(GROUPS)
    pf_count = len(_pf)
    return (
        "סטטוס בוט:\n"
        f"- שעה מקומית: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"- שעות שקטות: {quiet_desc} | עכשיו: {'פעיל' if quiet_now else 'לא'}\n"
        f"- קבוצות פעילות: {groups_count}\n"
        f"- מילות סינון: {pf_count} | מילים חוסמות: {block_desc}\n"
        f"- סריקה רצה: {'כן' if in_progress else 'לא'}\n"
        f"- סריקה אחרונה התחילה: {last_started.strftime('%Y-%m-%d %H:%M:%S %Z') if last_started else '-'}\n"
        f"- סריקה אחרונה הסתיימה: {last_finished.strftime('%Y-%m-%d %H:%M:%S %Z') if last_finished else '-'}\n"
        f"- סך נראו: {stats['seen']} | סך נשלחו: {stats['sent']}"
    )


def _build_daily_report_text(shared_state: dict) -> str:
    """בונה טקסט דו\"ח יומי — נתונים של היום בלבד."""
    now = _now_local()
    today_prefix = now.strftime("%Y-%m-%d")
    daily = get_daily_stats(today_prefix)
    total = get_stats()
    last_started = shared_state.get("last_scan_started")
    last_finished = shared_state.get("last_scan_finished")
    in_progress = bool(shared_state.get("scan_in_progress"))
    return (
        f"📋 דו\"ח יומי — {today_prefix}\n\n"
        f"- פוסטים שנסרקו היום: {daily['seen']}\n"
        f"- לידים שנשלחו היום: {daily['sent']}\n"
        f"- סריקה רצה כרגע: {'כן' if in_progress else 'לא'}\n"
        f"- סריקה אחרונה התחילה: {last_started.strftime('%H:%M:%S') if last_started else '-'}\n"
        f"- סריקה אחרונה הסתיימה: {last_finished.strftime('%H:%M:%S') if last_finished else '-'}\n\n"
        f"סה\"כ כללי: {total['seen']} נסרקו | {total['sent']} לידים"
    )


def _build_groups_text() -> str:
    """בונה טקסט רשימת קבוצות."""
    from scraper import GROUPS
    from database import _get_max_groups
    if GROUPS:
        lines = [f"{i}. {g['name']}\n   {g['url']}" for i, g in enumerate(GROUPS, 1)]
        header = f"קבוצות פעילות ({len(GROUPS)}"
        max_g = _get_max_groups()
        if max_g > 0:
            header += f"/{max_g}"
        header += "):"
        return header + "\n\n" + "\n\n".join(lines)
    return "אין קבוצות מוגדרות."


def _build_keywords_text() -> str:
    """בונה טקסט מילות מפתח לסינון + מילים חוסמות."""
    with _kw_lock:
        pf = _keywords_state["pre_filter"]
        bl = _keywords_state["block"]
    pf_text = ", ".join(pf) if pf else "(ריק)"
    bl_text = ", ".join(bl) if bl else "(ריק)"
    return (
        f"מילות מפתח לסינון ({len(pf)}):\n{pf_text}\n\n"
        f"מילים חוסמות ({len(bl)}):\n{bl_text}\n\n"
        "פקודות: /add_keyword <מילה> | /remove_keyword <מילה> | /block <מילה> | /unblock <מילה>"
    )


def _build_blocked_text() -> str:
    """בונה טקסט מילים חוסמות."""
    with _kw_lock:
        bl = _keywords_state["block"]
    bl_text = ", ".join(bl) if bl else "(ריק)"
    return (
        f"מילים חוסמות ({len(bl)}):\n{bl_text}\n\n"
        "פקודות: /block <מילה> | /unblock <מילה>"
    )


def _build_blocked_users_text() -> str:
    """בונה טקסט מפרסמים חסומים."""
    with _kw_lock:
        users = _keywords_state["blocked_users"]
    if users:
        lines = [f"• {u['name']}\n   {u['profile_url']}" for u in users]
        users_text = "\n\n".join(lines)
    else:
        users_text = "(ריק)"
    return (
        f"מפרסמים חסומים ({len(users)}):\n{users_text}\n\n"
        "פקודות: /block_user <URL פרופיל> | /unblock_user <URL פרופיל>"
    )


def _build_group_health_text() -> str:
    """בונה טקסט בריאות קבוצות — סטטיסטיקות לכל קבוצה."""
    from scraper import GROUPS
    health_data = get_all_group_health()
    if not health_data and not GROUPS:
        return "אין קבוצות מוגדרות."

    # מיפוי URL → שם ידידותי
    url_to_name: dict[str, str] = {g["url"]: g["name"] for g in GROUPS}

    # סף התראה — משמש גם לאייקונים
    try:
        _threshold = int(get_config("inactive_group_threshold", "50"))
    except (ValueError, TypeError):
        _threshold = 50
    if _threshold < 1:
        _threshold = 50
    # סף אזהרה = חצי מסף ההתראה (לפחות 1)
    _warn_threshold = max(1, _threshold // 2)

    lines = ["\U0001f4ca בריאות קבוצות:\n"]
    # קודם קבוצות עם health data
    seen_urls: set[str] = set()
    for h in health_data:
        url = h["group_url"]
        seen_urls.add(url)
        name = url_to_name.get(url, url.split("/groups/")[-1].rstrip("/") or url)
        consecutive = h["consecutive_empty"]
        last_count = h["last_posts_count"]
        last_lead = h["last_lead_at"]

        if consecutive >= _threshold:
            icon = "\u26a0\ufe0f"
        elif consecutive >= _warn_threshold:
            icon = "\U0001f7e1"
        else:
            icon = "\u2705"

        line = f"{icon} {name} — {last_count} פוסטים/סבב"
        if consecutive > 0:
            line += f" (0 כבר {consecutive} סבבים!)"
        if last_lead:
            # תצוגה ידידותית של last_lead_at
            try:
                from datetime import datetime as _dt
                lead_dt = _dt.fromisoformat(last_lead)
                now = _now_local()
                diff = now - lead_dt.replace(tzinfo=now.tzinfo) if lead_dt.tzinfo is None else now - lead_dt
                if diff.total_seconds() < 0:
                    lead_desc = "עכשיו"
                elif diff.days == 0:
                    hours = diff.seconds // 3600
                    if hours == 0:
                        lead_desc = "לפני כמה דקות"
                    else:
                        lead_desc = f"לפני {hours} שעות"
                elif diff.days == 1:
                    lead_desc = "אתמול"
                else:
                    lead_desc = f"לפני {diff.days} ימים"
            except Exception:
                lead_desc = last_lead[:10]
            line += f", ליד אחרון: {lead_desc}"
        else:
            line += ", ליד אחרון: -"
        lines.append(line)

    # קבוצות שעדיין לא נסרקו (אין health data)
    for g in GROUPS:
        if g["url"] not in seen_urls:
            lines.append(f"\u2753 {g['name']} — טרם נסרקה")

    return "\n".join(lines)


def _build_developer_usage_text() -> str:
    """בונה טקסט שימוש ב-API — נגיש רק למפתח."""
    now = _now_local()
    today_prefix = now.strftime("%Y-%m-%d")
    total = get_usage_stats()
    daily = get_daily_usage_stats(today_prefix)
    by_model = get_usage_by_model()

    lines = [
        "מעקב עלויות API:\n",
        f"היום ({today_prefix}):",
        f"  קריאות: {daily['total_calls']}",
        f"  טוקנים: {daily['total_tokens']:,} (in: {daily['prompt_tokens']:,} | out: {daily['completion_tokens']:,})",
        f"  עלות: ${daily['total_cost_usd']:.4f}\n",
        "סה\"כ מצטבר:",
        f"  קריאות: {total['total_calls']}",
        f"  טוקנים: {total['total_tokens']:,} (in: {total['prompt_tokens']:,} | out: {total['completion_tokens']:,})",
        f"  עלות: ${total['total_cost_usd']:.4f}",
    ]

    if by_model:
        lines.append("\nפירוט לפי מודל:")
        for m in by_model:
            lines.append(f"  {m['model']}: {m['total_calls']} קריאות, {m['total_tokens']:,} טוקנים, ${m['total_cost_usd']:.4f}")

    return "\n".join(lines)


def _parse_developer_chat_ids() -> set[int]:
    """מפרסר DEVELOPER_CHAT_ID (תומך ב-ID אחד או כמה מופרדים בפסיק)."""
    return _parse_allowed_chat_ids(DEVELOPER_CHAT_ID)


async def _handle_crud_command(
    text: str,
    chat_id: int,
    action_fn,
    usage_msg: str,
    *,
    reload_fn=None,
    pre_fn=None,
):
    """Helper לפקודות CRUD בטלגרם — מפרסר ארגומנט, מבצע פעולה, ושולח תשובה.

    action_fn: callable שמקבל מחרוזת ומחזיר (ok, msg).
    reload_fn: callable אופציונלי שייקרא אחרי הצלחה.
    pre_fn: callable אופציונלי שייקרא לפני הפעולה (למשל ensure_keywords_migrated).
    """
    from notifier import send_message

    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await asyncio.to_thread(
            send_message, usage_msg,
            chat_id=chat_id, disable_web_page_preview=True,
        )
        return
    arg = parts[1].strip()
    if pre_fn:
        pre_fn()
    ok, result_msg = action_fn(arg)
    if ok and reload_fn:
        reload_fn()
    await asyncio.to_thread(
        send_message, result_msg,
        chat_id=chat_id, disable_web_page_preview=True,
    )


async def _telegram_control_loop(
    *,
    scan_now_event: asyncio.Event,
    scan_force_event: asyncio.Event,
    shared_state: dict,
):
    """
    Telegram polling loop — תומך בפקודות טקסט ובכפתורי inline.
    /menu מציג תפריט עם כפתורים. כל פקודה ישנה ממשיכה לעבוד.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    allowed_chat_ids = _parse_allowed_chat_ids(os.environ.get("TELEGRAM_CHAT_ID"))
    if not bot_token or not allowed_chat_ids:
        log.info("בקרת טלגרם לא מופעלת (חסרים TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)")
        return

    from notifier import (
        send_message, send_message_with_buttons,
        edit_message_text, answer_callback_query,
    )
    import requests
    import json

    offset_file = Path(__file__).resolve().parent / "data" / "telegram_offset.txt"

    def load_offset() -> int:
        try:
            if offset_file.exists():
                return int(offset_file.read_text().strip() or "0")
        except Exception:
            return 0
        return 0

    def save_offset(value: int):
        try:
            offset_file.parent.mkdir(exist_ok=True)
            offset_file.write_text(str(value))
        except Exception:
            pass

    def fetch_updates(offset: int, timeout: int = 30) -> list[dict]:
        url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        params: dict = {"timeout": timeout, "allowed_updates": json.dumps(["message", "edited_message", "callback_query"])}
        if offset:
            params["offset"] = offset
        resp = requests.get(url, params=params, timeout=timeout + 10)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram getUpdates failed: {data}")
        return data.get("result") or []

    # ── טיפול ב-callback query (לחיצה על כפתור) ──────────────
    async def handle_callback(cb: dict):
        cb_id = cb.get("id", "")
        cb_data = cb.get("data", "")
        cb_msg = cb.get("message") or {}
        chat_id = cb_msg.get("chat", {}).get("id")
        message_id = cb_msg.get("message_id")

        if chat_id is None:
            return
        try:
            chat_id_int = int(chat_id)
        except Exception:
            return
        if chat_id_int not in allowed_chat_ids:
            return

        # עונים לטלגרם כדי להסיר אנימציית טעינה
        await asyncio.to_thread(answer_callback_query, cb_id)

        if cb_data == "menu":
            await asyncio.to_thread(
                edit_message_text, chat_id_int, message_id,
                "תפריט ראשי — בחר פעולה:", _main_menu_buttons(),
            )

        elif cb_data == "scan":
            now = _now_local()
            if shared_state.get("scan_in_progress"):
                await asyncio.to_thread(
                    edit_message_text, chat_id_int, message_id,
                    "כבר יש סריקה שרצה כרגע.",
                    _back_to_menu_button(),
                )
                return
            # cooldown דקה מסיום סריקה אחרונה
            last_finished = shared_state.get("last_scan_finished")
            if last_finished:
                elapsed = (now - last_finished).total_seconds()
                if elapsed < 60:
                    remaining = int(60 - elapsed)
                    await asyncio.to_thread(
                        edit_message_text, chat_id_int, message_id,
                        f"סריקה הסתיימה לפני {int(elapsed)} שניות. נסה שוב בעוד {remaining} שניות.",
                        _back_to_menu_button(),
                    )
                    return
            quiet = shared_state.get("quiet")
            if quiet and _is_quiet_now(now, quiet):
                qh_db = get_config("quiet_hours", _CONFIG_NOT_SET)
                quiet_spec = qh_db if qh_db is not _CONFIG_NOT_SET else QUIET_HOURS
                await asyncio.to_thread(
                    edit_message_text, chat_id_int, message_id,
                    f"שעות שקטות פעילות עכשיו ({quiet_spec}, {TIMEZONE_NAME}). השתמש /scan_force לסריקה בכל זאת.",
                    _back_to_menu_button(),
                )
                return
            scan_now_event.set()
            await asyncio.to_thread(
                edit_message_text, chat_id_int, message_id,
                "מתזמן סריקה עכשיו.",
                _back_to_menu_button(),
            )

        elif cb_data == "status":
            status_text = _build_status_text(shared_state)
            await asyncio.to_thread(
                edit_message_text, chat_id_int, message_id,
                status_text, _back_to_menu_button(),
            )

        elif cb_data == "daily_report":
            report_text = _build_daily_report_text(shared_state)
            await asyncio.to_thread(
                edit_message_text, chat_id_int, message_id,
                report_text, _back_to_menu_button(),
            )

        elif cb_data == "settings":
            await asyncio.to_thread(
                edit_message_text, chat_id_int, message_id,
                "הגדרות — בחר קטגוריה:", _settings_menu_buttons(),
            )

        elif cb_data == "groups":
            groups_text = _build_groups_text()
            await asyncio.to_thread(
                edit_message_text, chat_id_int, message_id,
                groups_text,
                [[{"text": "\U0001f519 חזרה להגדרות", "callback_data": "settings"}]],
            )

        elif cb_data == "keywords":
            kw_text = _build_keywords_text()
            await asyncio.to_thread(
                edit_message_text, chat_id_int, message_id,
                kw_text,
                [[{"text": "\U0001f519 חזרה להגדרות", "callback_data": "settings"}]],
            )

        elif cb_data == "blocked":
            bl_text = _build_blocked_text()
            await asyncio.to_thread(
                edit_message_text, chat_id_int, message_id,
                bl_text,
                [[{"text": "\U0001f519 חזרה להגדרות", "callback_data": "settings"}]],
            )

        elif cb_data == "blocked_users":
            bu_text = _build_blocked_users_text()
            await asyncio.to_thread(
                edit_message_text, chat_id_int, message_id,
                bu_text,
                [[{"text": "\U0001f519 חזרה להגדרות", "callback_data": "settings"}]],
            )

    # ניקוי webhook קודם וביטול polling תקוע מאינסטנס ישן
    try:
        await asyncio.to_thread(
            requests.post,
            f"https://api.telegram.org/bot{bot_token}/deleteWebhook",
            json={"drop_pending_updates": False},
            timeout=10,
        )
        log.debug("webhook נוקה בהצלחה")
    except Exception:
        pass

    # רישום פקודות בוט בטלגרם — כפתור "/" יציג את הפקודות הזמינות
    try:
        await asyncio.to_thread(
            requests.post,
            f"https://api.telegram.org/bot{bot_token}/setMyCommands",
            json={"commands": [
                {"command": "menu", "description": "תפריט ראשי עם כפתורים"},
                {"command": "scan", "description": "סריקה עכשיו"},
                {"command": "status", "description": "סטטוס הבוט"},
                {"command": "groups", "description": "רשימת קבוצות"},
                {"command": "keywords", "description": "מילות מפתח"},
                {"command": "force_send", "description": "מילות 'שלח תמיד' (עוקף AI)"},
                {"command": "unforce", "description": "הסרת מילת 'שלח תמיד'"},
                {"command": "hot_word", "description": "מילים חמות (ליד חם עם התראה)"},
                {"command": "unhot", "description": "הסרת מילה חמה"},
                {"command": "block", "description": "הוספת מילה חוסמת"},
                {"command": "unblock", "description": "הסרת מילה חוסמת"},
                {"command": "block_user", "description": "חסימת מפרסם (רשימה שחורה)"},
                {"command": "unblock_user", "description": "הסרת חסימת מפרסם"},
                {"command": "blocked_users", "description": "רשימת מפרסמים חסומים"},
                {"command": "max_age", "description": "סינון פוסטים ישנים (לפי ימים)"},
                {"command": "health", "description": "בריאות קבוצות — סטטיסטיקות"},
                {"command": "panel", "description": "לינק לפאנל ניהול"},
            ]},
            timeout=10,
        )
        log.debug("פקודות בוט נרשמו בטלגרם")
    except Exception:
        pass

    offset = load_offset()
    log.info("בקרת טלגרם הופעלה. פקודות: /menu /scan /status")
    conflict_backoff = 5

    while True:
        try:
            updates = await asyncio.to_thread(fetch_updates, offset, 30)
            conflict_backoff = 5  # איפוס backoff אחרי הצלחה
            for upd in updates:
                try:
                    update_id = int(upd.get("update_id"))
                except Exception:
                    continue

                offset = max(offset, update_id + 1)
                save_offset(offset)

                # ── callback query (לחיצה על כפתור) ──────────
                cb = upd.get("callback_query")
                if cb:
                    await handle_callback(cb)
                    continue

                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue

                chat_id = msg.get("chat", {}).get("id")
                if chat_id is None:
                    continue

                try:
                    chat_id_int = int(chat_id)
                except Exception:
                    continue

                if chat_id_int not in allowed_chat_ids:
                    continue

                text = (msg.get("text") or "").strip()
                if not text:
                    continue

                first = text.split()[0].strip()
                cmd = first.split("@", 1)[0].lower()

                if cmd in ("/menu", "/start", "/help"):
                    await asyncio.to_thread(
                        send_message_with_buttons,
                        "תפריט ראשי — בחר פעולה:",
                        _main_menu_buttons(),
                        chat_id=chat_id_int,
                    )
                    continue

                if cmd in ("/status",):
                    status_text = _build_status_text(shared_state)
                    await asyncio.to_thread(
                        send_message_with_buttons,
                        status_text,
                        _back_to_menu_button(),
                        chat_id=chat_id_int,
                        disable_web_page_preview=True,
                    )
                    continue

                if cmd in ("/panel",):
                    if PANEL_URL:
                        await asyncio.to_thread(
                            send_message,
                            f"🔗 פאנל הגדרות:\n{PANEL_URL}",
                            chat_id=chat_id_int,
                        )
                    else:
                        await asyncio.to_thread(
                            send_message,
                            "משתנה סביבה PANEL_URL לא הוגדר.",
                            chat_id=chat_id_int,
                        )
                    continue

                if cmd in ("/scan", "/scan_force"):
                    is_force = cmd == "/scan_force"
                    now = _now_local()

                    # הגנה 1: סריקה כבר רצה כרגע
                    if shared_state.get("scan_in_progress"):
                        await asyncio.to_thread(
                            send_message,
                            "כבר יש סריקה שרצה כרגע.",
                            chat_id=chat_id_int,
                            disable_web_page_preview=True,
                        )
                        continue

                    # הגנה 2: cooldown דקה מסיום סריקה אחרונה
                    last_finished = shared_state.get("last_scan_finished")
                    if last_finished:
                        elapsed = (now - last_finished).total_seconds()
                        if elapsed < 60:
                            remaining = int(60 - elapsed)
                            await asyncio.to_thread(
                                send_message,
                                f"סריקה הסתיימה לפני {int(elapsed)} שניות. נסה שוב בעוד {remaining} שניות.",
                                chat_id=chat_id_int,
                                disable_web_page_preview=True,
                            )
                            continue

                    # בדיקת שעות שקטות (רק ל-/scan רגיל)
                    if not is_force:
                        quiet = shared_state.get("quiet")
                        qh_db = get_config("quiet_hours", _CONFIG_NOT_SET)
                        quiet_spec = qh_db if qh_db is not _CONFIG_NOT_SET else QUIET_HOURS
                        if quiet and _is_quiet_now(now, quiet):
                            await asyncio.to_thread(
                                send_message,
                                f"שעות שקטות פעילות עכשיו ({quiet_spec}, {TIMEZONE_NAME}). אם בכל זאת צריך, השתמש /scan_force.",
                                chat_id=chat_id_int,
                                disable_web_page_preview=True,
                            )
                            continue

                    if is_force:
                        scan_force_event.set()
                    scan_now_event.set()
                    label = "forced" if is_force else "רגילה"
                    await asyncio.to_thread(
                        send_message,
                        f"מתזמן סריקה עכשיו ({label}).",
                        chat_id=chat_id_int,
                        disable_web_page_preview=True,
                    )
                    continue

                if cmd in ("/stop",):
                    dev_ids = _parse_developer_chat_ids()
                    if not dev_ids or chat_id_int not in dev_ids:
                        await asyncio.to_thread(
                            send_message, "פקודה זו זמינה רק למפתח (DEVELOPER_CHAT_ID).",
                            chat_id=chat_id_int,
                        )
                        continue
                    if not shared_state.get("scan_in_progress"):
                        await asyncio.to_thread(
                            send_message, "אין סריקה פעילה כרגע.",
                            chat_id=chat_id_int,
                        )
                        continue
                    from scraper import request_stop_scan
                    request_stop_scan()
                    await asyncio.to_thread(
                        send_message,
                        "ביקשתי עצירת סריקה — תיעצר אחרי הקבוצה הנוכחית.",
                        chat_id=chat_id_int,
                    )
                    continue

                # ── ניהול קבוצות דינמי ──────────────────────────
                if cmd in ("/groups",):
                    msg_text = _build_groups_text()
                    await asyncio.to_thread(
                        send_message_with_buttons, msg_text,
                        _back_to_menu_button(),
                        chat_id=chat_id_int,
                        disable_web_page_preview=True,
                    )
                    continue

                if cmd in ("/add_group",):
                    from scraper import reload_groups
                    await _handle_crud_command(
                        text, chat_id_int, add_group,
                        "שימוש: /add_group <url>",
                        reload_fn=reload_groups,
                    )
                    continue

                if cmd in ("/remove_group",):
                    from scraper import reload_groups
                    await _handle_crud_command(
                        text, chat_id_int, remove_group,
                        "שימוש: /remove_group <url>",
                        reload_fn=reload_groups,
                    )
                    continue

                # ── ניהול מילות מפתח דינמי ──────────────────────
                if cmd in ("/keywords",):
                    msg_text = _build_keywords_text()
                    await asyncio.to_thread(
                        send_message_with_buttons, msg_text,
                        _back_to_menu_button(),
                        chat_id=chat_id_int,
                        disable_web_page_preview=True,
                    )
                    continue

                if cmd in ("/add_keyword",):
                    await _handle_crud_command(
                        text, chat_id_int,
                        lambda w: add_keyword(w, "pre_filter"),
                        "שימוש: /add_keyword <מילה>",
                        reload_fn=reload_keywords,
                        pre_fn=lambda: ensure_keywords_migrated("pre_filter", _PRE_FILTER_KEYWORDS_DEFAULT),
                    )
                    continue

                if cmd in ("/remove_keyword",):
                    await _handle_crud_command(
                        text, chat_id_int,
                        lambda w: remove_keyword(w, "pre_filter"),
                        "שימוש: /remove_keyword <מילה>",
                        reload_fn=reload_keywords,
                    )
                    continue

                if cmd in ("/block",):
                    await _handle_crud_command(
                        text, chat_id_int,
                        lambda w: add_keyword(w, "block"),
                        "שימוש: /block <מילה>",
                        reload_fn=reload_keywords,
                        pre_fn=lambda: ensure_keywords_migrated("block", _BLOCK_KEYWORDS_ENV),
                    )
                    continue

                if cmd in ("/unblock",):
                    await _handle_crud_command(
                        text, chat_id_int,
                        lambda w: remove_keyword(w, "block"),
                        "שימוש: /unblock <מילה>",
                        reload_fn=reload_keywords,
                    )
                    continue

                # ── ניהול רשימה שחורה של מפרסמים ──────────────
                if cmd in ("/block_user",):
                    await _handle_crud_command(
                        text, chat_id_int,
                        add_blocked_user,
                        "שימוש: /block_user <URL פרופיל פייסבוק>",
                        reload_fn=reload_keywords,
                    )
                    continue

                if cmd in ("/unblock_user",):
                    await _handle_crud_command(
                        text, chat_id_int,
                        remove_blocked_user,
                        "שימוש: /unblock_user <URL פרופיל פייסבוק>",
                        reload_fn=reload_keywords,
                    )
                    continue

                if cmd in ("/blocked_users",):
                    msg_text = _build_blocked_users_text()
                    await asyncio.to_thread(
                        send_message_with_buttons, msg_text,
                        _back_to_menu_button(),
                        chat_id=chat_id_int,
                        disable_web_page_preview=True,
                    )
                    continue

                # ── "שלח תמיד" — עוקף סיווג AI ──────────────────
                if cmd in ("/force_send",):
                    parts = text.split(maxsplit=1)
                    if len(parts) < 2 or not parts[1].strip():
                        with _kw_lock:
                            fs = _keywords_state["force_send"]
                        fs_text = ", ".join(fs) if fs else "(ריק)"
                        await asyncio.to_thread(
                            send_message,
                            f"מילות 'שלח תמיד' ({len(fs)}):\n{fs_text}\n\n"
                            "שימוש: /force_send <מילה> — פוסט שמכיל מילה זו יישלח ישירות ללא AI",
                            chat_id=chat_id_int, disable_web_page_preview=True,
                        )
                        continue
                    ok, result_msg = add_force_send_keyword(parts[1].strip())
                    if ok:
                        reload_keywords()
                    await asyncio.to_thread(
                        send_message, result_msg,
                        chat_id=chat_id_int, disable_web_page_preview=True,
                    )
                    continue

                if cmd in ("/unforce",):
                    await _handle_crud_command(
                        text, chat_id_int,
                        remove_force_send_keyword,
                        "שימוש: /unforce <מילה>",
                        reload_fn=reload_keywords,
                    )
                    continue

                # ── מילים חמות — ליד חם עם התראה מיוחדת ────────
                if cmd in ("/hot_word",):
                    parts = text.split(maxsplit=1)
                    if len(parts) < 2 or not parts[1].strip():
                        with _kw_lock:
                            hw = _keywords_state.get("hot_words", [])
                        hw_text = ", ".join(hw) if hw else "(ריק)"
                        await asyncio.to_thread(
                            send_message,
                            f"\U0001f525 מילים חמות ({len(hw)}):\n{hw_text}\n\n"
                            "שימוש: /hot_word <מילה> — פוסט שמכיל מילה זו יסומן כליד חם\n"
                            "/unhot <מילה> — הסרת מילה חמה",
                            chat_id=chat_id_int, disable_web_page_preview=True,
                        )
                        continue
                    ok, result_msg = add_hot_word(parts[1].strip())
                    if ok:
                        reload_keywords()
                    await asyncio.to_thread(
                        send_message, result_msg,
                        chat_id=chat_id_int, disable_web_page_preview=True,
                    )
                    continue

                if cmd in ("/unhot",):
                    await _handle_crud_command(
                        text, chat_id_int,
                        remove_hot_word,
                        "שימוש: /unhot <מילה>",
                        reload_fn=reload_keywords,
                    )
                    continue

                # ── גיל מקסימלי לפוסטים ────────────────────────
                if cmd in ("/max_age",):
                    parts = text.split(maxsplit=1)
                    if len(parts) < 2 or not parts[1].strip():
                        current = _load_max_post_age()
                        status = f"{current} ימים" if current else "כבוי (ללא הגבלה)"
                        await asyncio.to_thread(
                            send_message,
                            f"סינון פוסטים ישנים: {status}\n\n"
                            "שימוש: /max_age <ימים> — פוסטים ישנים מהמספר הזה לא יישלחו\n"
                            "/max_age 0 — ביטול סינון לפי גיל",
                            chat_id=chat_id_int, disable_web_page_preview=True,
                        )
                        continue
                    arg = parts[1].strip()
                    try:
                        days = int(arg)
                    except ValueError:
                        await asyncio.to_thread(
                            send_message, "יש להזין מספר שלם (למשל: /max_age 2)",
                            chat_id=chat_id_int, disable_web_page_preview=True,
                        )
                        continue
                    if days <= 0:
                        set_config("max_post_age_days", "0")
                        await asyncio.to_thread(
                            send_message, "סינון לפי גיל בוטל — כל הפוסטים יישלחו.",
                            chat_id=chat_id_int, disable_web_page_preview=True,
                        )
                    else:
                        set_config("max_post_age_days", str(days))
                        await asyncio.to_thread(
                            send_message,
                            f"סינון גיל הוגדר: פוסטים ישנים מ-{days} ימים לא יישלחו.",
                            chat_id=chat_id_int, disable_web_page_preview=True,
                        )
                    continue

                # ── בריאות קבוצות — למפתח בלבד ──────────────
                if cmd in ("/health",):
                    dev_ids = _parse_developer_chat_ids()
                    if not dev_ids or chat_id_int not in dev_ids:
                        await asyncio.to_thread(
                            send_message, "פקודה זו זמינה רק למפתח (DEVELOPER_CHAT_ID).",
                            chat_id=chat_id_int, disable_web_page_preview=True,
                        )
                        continue
                    health_text = _build_group_health_text()
                    await asyncio.to_thread(
                        send_message_with_buttons,
                        health_text,
                        _back_to_menu_button(),
                        chat_id=chat_id_int,
                        disable_web_page_preview=True,
                    )
                    continue

                # ── מעקב עלויות — למפתח בלבד ─────────────────
                if cmd in ("/developer_usage",):
                    dev_ids = _parse_developer_chat_ids()
                    if not dev_ids or chat_id_int not in dev_ids:
                        await asyncio.to_thread(
                            send_message, "פקודה זו זמינה רק למפתח.",
                            chat_id=chat_id_int, disable_web_page_preview=True,
                        )
                        continue
                    usage_text = _build_developer_usage_text()
                    await asyncio.to_thread(
                        send_message_with_buttons,
                        usage_text,
                        _back_to_menu_button(),
                        chat_id=chat_id_int,
                        disable_web_page_preview=True,
                    )
                    continue

                # ── Debug קבוצה בודדת — למפתח בלבד ────────────
                if cmd in ("/debug",):
                    dev_ids = _parse_developer_chat_ids()
                    if not dev_ids or chat_id_int not in dev_ids:
                        await asyncio.to_thread(
                            send_message, "פקודה זו זמינה רק למפתח (DEVELOPER_CHAT_ID).",
                            chat_id=chat_id_int, disable_web_page_preview=True,
                        )
                        continue
                    parts = text.split(maxsplit=1)
                    if len(parts) < 2 or not parts[1].strip():
                        await asyncio.to_thread(
                            send_message,
                            "שימוש: /debug <group_url>\nסורק קבוצה אחת ומציג את כל שלבי העיבוד.",
                            chat_id=chat_id_int, disable_web_page_preview=True,
                        )
                        continue
                    debug_url = parts[1].strip()
                    await asyncio.to_thread(
                        send_message, f"מתחיל סריקת debug לקבוצה:\n{debug_url}",
                        chat_id=chat_id_int, disable_web_page_preview=True,
                    )
                    try:
                        debug_result = await _run_debug_scan(debug_url)
                        await asyncio.to_thread(
                            send_message, debug_result,
                            chat_id=chat_id_int, disable_web_page_preview=True,
                        )
                    except Exception as e:
                        await asyncio.to_thread(
                            send_message, f"שגיאה בסריקת debug:\n{e}",
                            chat_id=chat_id_int, disable_web_page_preview=True,
                        )
                    continue


        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 409:
                # 409 Conflict = אינסטנס אחר עושה polling — ממתין עם backoff
                log.warning(f"409 Conflict — אינסטנס אחר פעיל. ממתין {conflict_backoff} שניות...")
                await asyncio.sleep(conflict_backoff)
                conflict_backoff = min(conflict_backoff * 2, 120)
            else:
                log.error(f"שגיאה בלולאת בקרת טלגרם: {e}", exc_info=True)
                await asyncio.sleep(5)
        except Exception as e:
            log.error(f"שגיאה בלולאת בקרת טלגרם: {e}", exc_info=True)
            await asyncio.sleep(5)

# רפרנס ל-shared_state — מאותחל ב-main() כדי ש-health check יוכל לגשת למצב הסריקה
_health_shared_state: dict | None = None
# סף מקסימלי (בדקות) מאז סריקה אחרונה לפני שה-health check מדווח על בעיה
_HEALTH_MAX_SCAN_AGE_MINUTES = int(os.environ.get("HEALTH_MAX_SCAN_AGE_MINUTES",
                                                   str(INTERVAL_MINUTES * 3 + 5)))


def _deep_health_check() -> tuple[int, dict]:
    """בודק תקינות מעמיקה. מחזיר (status_code, details_dict)."""
    import shutil
    from scraper import SESSION_FILE
    from database import DB_PATH

    checks = {}
    healthy = True

    # 1. בדיקת DB
    try:
        get_stats()
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"error: {e}"
        healthy = False

    # 2. בדיקת session file
    session_exists = SESSION_FILE.exists()
    checks["session_file"] = "ok" if session_exists else "missing"
    if not session_exists:
        healthy = False

    # 3. זמן מאז סריקה אחרונה
    state = _health_shared_state
    if state is not None:
        last_finished = state.get("last_scan_finished")
        in_progress = state.get("scan_in_progress", False)
        if last_finished:
            age = (_now_local() - last_finished).total_seconds() / 60
            checks["last_scan_age_minutes"] = round(age, 1)
            if age > _HEALTH_MAX_SCAN_AGE_MINUTES and not in_progress:
                checks["last_scan"] = "stale"
                healthy = False
            else:
                checks["last_scan"] = "ok"
        elif in_progress:
            checks["last_scan"] = "first_scan_running"
        else:
            checks["last_scan"] = "no_scan_yet"
    else:
        checks["last_scan"] = "not_initialized"

    # 4. בדיקת מקום בדיסק
    try:
        usage = shutil.disk_usage(DB_PATH.parent)
        free_mb = usage.free / (1024 * 1024)
        checks["disk_free_mb"] = round(free_mb, 1)
        if free_mb < 50:
            checks["disk"] = "low"
            healthy = False
        else:
            checks["disk"] = "ok"
    except Exception as e:
        checks["disk"] = f"error: {e}"
        healthy = False

    status_code = 200 if healthy else 503
    checks["status"] = "healthy" if healthy else "unhealthy"
    return status_code, checks


class _HealthHandler(BaseHTTPRequestHandler):
    """Healthcheck endpoint — GET / מחזיר 200 OK, GET /health מחזיר בדיקה מעמיקה."""

    def do_GET(self):
        if self.path == "/health":
            status_code, details = _deep_health_check()
            import json as _json
            body = _json.dumps(details, ensure_ascii=False).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

    def log_message(self, format, *args):
        # השתקת לוג של http.server כדי לא להציף את הלוגים
        pass


def start_health_server(port: int = HEALTH_PORT):
    """מפעיל שרת HTTP פשוט לבדיקת חיוּת על daemon thread."""
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    log.info(f"Health server רץ על פורט {port}")
    return server


def _load_interval_from_db() -> int:
    """מחזיר תדירות סריקה ממשתנה סביבה בלבד.
    interval_minutes כבר לא ניתן לעריכה מהפאנל — רק דרך env var.
    """
    return max(1, INTERVAL_MINUTES)

_CONFIG_NOT_SET = object()

def _load_quiet_hours_from_db() -> tuple[time, time] | None:
    """טוען שעות שקטות מ-DB עם fallback למשתנה סביבה.
    ערך ריק ב-DB = המשתמש ביטל שעות שקטות דרך הפאנל — מחזיר None.
    """
    db_val = get_config("quiet_hours", _CONFIG_NOT_SET)
    if db_val is not _CONFIG_NOT_SET:
        # יש ערך ב-DB (גם אם ריק) — הפאנל קובע
        spec = db_val.strip()
    else:
        # אין ערך ב-DB — fallback למשתנה סביבה
        spec = QUIET_HOURS
    if not spec:
        return None
    try:
        return _parse_quiet_hours(spec)
    except Exception as e:
        log.error(f"QUIET_HOURS לא תקין ({spec}) — מתעלם. שגיאה: {e}")
        return None

def _get_panel_port() -> int:
    """מחזיר את הפורט שהפאנל ישתמש בו."""
    return int(os.environ.get("PANEL_PORT", os.environ.get("PORT", "8080")))


def _start_panel() -> bool:
    """מפעיל פאנל הגדרות כ-daemon thread (אם Flask מותקן).
    מחזיר True רק אם Flask הצליח לעשות bind לפורט.
    """
    panel_port = _get_panel_port()
    try:
        from panel import create_app
        app = create_app()
        # app.run() חוסם כשהשרת רץ. אם הוא נופל (port conflict וכו'),
        # ה-thread יוצא ו-bind_failed מסומן. ממתינים עד שנייה לוודא bind הצליח.
        bind_failed = threading.Event()
        bind_error: list = [None]

        def run():
            try:
                app.run(host="0.0.0.0", port=panel_port, debug=False, use_reloader=False)
            except Exception as e:
                bind_error[0] = e
            finally:
                bind_failed.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        # אם app.run() הצליח, הוא חוסם לנצח ו-bind_failed לעולם לא מסומן.
        # אם נכשל, ה-thread יוצא מהר ו-bind_failed מסומן.
        if bind_failed.wait(timeout=1.0):
            log.warning(f"פאנל הגדרות נכשל בהפעלה: {bind_error[0]}")
            return False
        log.info(f"פאנל הגדרות רץ על פורט {panel_port}")
        return True
    except ImportError:
        log.debug("Flask לא מותקן — פאנל הגדרות לא יופעל")
        return False
    except Exception as e:
        log.warning(f"לא ניתן להפעיל פאנל הגדרות: {e}")
        return False


_shutting_down = False

def handle_signal(sig, frame):
    global _shutting_down
    log.info(f"קיבלנו סיגנל {sig} — יוצאים בצורה מסודרת")
    _shutting_down = True
    sys.exit(0)

# רישום signal handlers רק ב-main thread — כשהמודול מיובא מ-thread אחר
# (למשל Flask panel) signal.signal() זורק ValueError
if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

def _content_dedup_hash(text: str) -> str:
    """מחשב hash תוכן לזיהוי כפילויות — מבוסס על ליבת הטקסט בלבד.

    משתמש ב-12 המילים הראשונות של הטקסט המנורמל (אחרי _stable_text_for_hash).
    הסיבה: innerText של אלמנט פוסט כולל גם תגובות/תשובות שמופיעים בסוף הטקסט.
    בין סריקות, תגובות חדשות מתווספות → ה-hash של הטקסט המלא משתנה → כפילות.
    ליבת הפוסט (שם הכותב + תחילת הגוף) מופיעה תמיד בהתחלה ולא משתנה.

    למה 12 מילים ולא 150 תווים: בפוסטים קצרים (~19 מילים / ~100 תווים) חיתוך
    לפי תווים כולל חלק מתגובות — שמשתנות בין סריקות. 12 מילים = ~60-80 תווים
    של ליבת הפוסט (שם + משפט ראשון), וספירת מילים לא מושפעת מאורך הפוסט.
    """
    import hashlib
    from scraper import _stable_text_for_hash
    stable = _stable_text_for_hash(text)
    if not stable:
        return ""
    # 12 מילים ראשונות — ליבת הפוסט (שם + משפט ראשון) יציבה בין סריקות.
    # מילים עדיפות על תווים כי תוכן דינמי (תגובות) מתחיל אחרי גוף הפוסט,
    # וספירת מילים לא מושפעת מאורכי שורות או שינויי רווחים.
    # 12 ולא 20 כי בפוסטים קצרים (~19 מילים) חיתוך ל-20 כולל חלק מתגובות.
    words = stable.split()[:12]
    core = ' '.join(words)
    return hashlib.md5(core.encode()).hexdigest()

def _dedup_debug(post_id: str, content: str, c_hash: str, label: str):
    """לוג דיאגנוסטי לזיהוי כפילויות — מציג snippet + hash + post_id + stable text."""
    snippet = content.replace('\n', ' ')[:80]
    log.info(f"[DEDUP-SEND] {label} | id={post_id[:20]} | hash={c_hash[:12]} | \"{snippet}...\"")
    # לוג מורחב: מציג את הטקסט המנורמל המלא (אחרי _stable_text_for_hash)
    # זה מאפשר להשוות בין סריקות ולראות מה בדיוק שונה כשה-hash לא תואם
    from scraper import _stable_text_for_hash
    stable = _stable_text_for_hash(content)
    log.debug(f"[DEDUP-STABLE] {c_hash[:12]} | stable_len={len(stable)} | \"{stable[:300]}...\"")

def _reset_scan_progress():
    """מאפס את scan_progress לפני סבב חדש."""
    scan_progress.update({
        "active": True,
        "phase": "scraping",
        "phase_label": "סורק קבוצות",
        "current_group": "",
        "current_group_index": 0,
        "total_groups": 0,
        "groups_done": [],
        "total_posts_found": 0,
        "new_posts": 0,
        "posts_to_classify": 0,
        "leads_sent": 0,
        "started_at": _now_local().isoformat(),
        "finished_at": None,
        "error": "",
    })


def _finish_scan_progress(leads: int, error: str = ""):
    """מסמן סיום סריקה ב-scan_progress."""
    scan_progress.update({
        "active": False,
        "phase": "done" if not error else "error",
        "phase_label": "הסתיים" if not error else "שגיאה",
        "leads_sent": leads,
        "finished_at": _now_local().isoformat(),
        "error": error,
    })


def _on_group_scraped(group_name: str, group_index: int, total_groups: int, posts_found: int):
    """callback שנקרא מ-scrape_all אחרי כל קבוצה — מעדכן את scan_progress."""
    scan_progress["current_group"] = group_name
    scan_progress["current_group_index"] = group_index
    scan_progress["total_groups"] = total_groups
    done_list = scan_progress["groups_done"]
    done_list.append({"name": group_name, "posts_found": posts_found})
    scan_progress["groups_done"] = done_list
    scan_progress["total_posts_found"] = sum(g["posts_found"] for g in done_list)


async def _run_debug_scan(group_url: str) -> str:
    """סורק קבוצה בודדת במצב debug ומחזיר דו\"ח מפורט של כל שלבי העיבוד."""
    import gc
    from scraper import scrape_group, GROUPS, load_session, save_session, login
    from scraper import block_heavy_resources, random_delay, dismiss_cookie_dialog
    from scraper import _is_wui_page, _has_login_overlay
    from database import _normalize_group_url, DB_PATH

    normalized_url = _normalize_group_url(group_url)
    # חיפוש שם הקבוצה ברשימה
    group_name = normalized_url.split("/groups/")[-1].rstrip("/") or normalized_url
    for g in GROUPS:
        if g["url"] == normalized_url:
            group_name = g["name"]
            break

    group = {"name": group_name, "url": normalized_url}
    lines = [f"\U0001f50d דו\"ח Debug: {group_name}\n{normalized_url}\n"]

    # שלב 1 — סריקה
    fb_email = get_config("fb_email") or FB_EMAIL
    fb_password = get_config_encrypted("fb_password") or FB_PASSWORD
    if not fb_email or not fb_password:
        return "חסרים FB_EMAIL / FB_PASSWORD — לא ניתן לסרוק."

    from playwright.async_api import async_playwright
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                       "--renderer-process-limit=1", "--js-flags=--max-old-space-size=128"]
            )
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
                viewport={"width": 360, "height": 640},
                locale="he-IL"
            )
            page = await context.new_page()
            await block_heavy_resources(page)

            session_loaded = await load_session(context)
            if session_loaded:
                await page.goto("https://m.facebook.com", wait_until="domcontentloaded")
                await random_delay(1, 2)
                await dismiss_cookie_dialog(page)
                if "login" in page.url or await _has_login_overlay(page) or await _is_wui_page(page):
                    await context.clear_cookies()
                    await login(page, email=fb_email, password=fb_password, browser=browser)
                    await save_session(context)
            else:
                await login(page, email=fb_email, password=fb_password, browser=browser)
                await save_session(context)

            # ללא seen_checker — כדי לא לעצור גלילה מוקדם ולקבל מקסימום פוסטים
            posts = await scrape_group(page, group, seen_checker=None)
            await browser.close()
    except Exception as e:
        return f"שגיאה בסריקה: {e}"

    gc.collect()

    lines.append(f"\U0001f4e6 פוסטים שנטענו מהקבוצה: {len(posts)}")

    # שלב 2 — כמה חדשים (בדיקת seen בנפרד — כי לא העברנו seen_checker לסקריפר)
    new_posts = [p for p in posts if not is_seen(p["id"])]
    already_seen = len(posts) - len(new_posts)
    lines.append(f"\U0001f195 פוסטים חדשים (לא נראו): {len(new_posts)}")
    if already_seen:
        lines.append(f"   כבר נראו בעבר: {already_seen}")

    # שלב 2.5 — content_hash dedup
    dedup_caught = 0
    after_dedup = []
    for p in new_posts:
        c_hash = _content_dedup_hash(p["content"])
        if c_hash and is_content_hash_sent(c_hash):
            dedup_caught += 1
            continue
        after_dedup.append(p)
    if dedup_caught:
        lines.append(f"\U0001f503 כפילויות לפי content_hash: {dedup_caught}")

    # שלב 3 — סינון גיל
    max_age = _load_max_post_age()
    age_filtered = 0
    if max_age:
        before = len(after_dedup)
        after_dedup = [p for p in after_dedup if not is_post_too_old(p["content"], max_age)]
        age_filtered = before - len(after_dedup)
    if age_filtered:
        lines.append(f"\u23f3 סוננו לפי גיל ({max_age} ימים): {age_filtered}")

    # שלב 4 — סינון מפרסמים חסומים
    user_blocked = 0
    with _kw_lock:
        _has_bu = bool(_keywords_state["blocked_users"])
    if _has_bu:
        before = len(after_dedup)
        after_dedup = [p for p in after_dedup if not is_user_blocked(p.get("author_url", ""))]
        user_blocked = before - len(after_dedup)
    if user_blocked:
        lines.append(f"\U0001f6d1 נחסמו לפי מפרסם: {user_blocked}")

    # שלב 5 — סינון מילים חוסמות
    blocked_count = 0
    after_block = []
    for p in after_dedup:
        if is_blocked(p["content"]):
            blocked_count += 1
        else:
            after_block.append(p)
    if blocked_count:
        lines.append(f"\U0001f6ab נחסמו לפי מילים חוסמות: {blocked_count}")

    # שלב 6 — סינון מילות מפתח
    passed_filter = [p for p in after_block if passes_keyword_filter(p["content"])]
    kw_filtered = len(after_block) - len(passed_filter)
    lines.append(f"\U0001f50d עברו סינון מילות מפתח: {len(passed_filter)}")
    if kw_filtered:
        lines.append(f"   סוננו (לא מכילים מילת מפתח): {kw_filtered}")

    # שלב 7 — כמה כבר נשלחו
    not_sent = [p for p in passed_filter if not is_lead_sent(p["id"])]
    already_sent = len(passed_filter) - len(not_sent)
    lines.append(f"\u2709\ufe0f ממתינים לסיווג AI: {len(not_sent)}")
    if already_sent:
        lines.append(f"   כבר נשלחו: {already_sent}")

    # דוגמה לפוסט שנפסל (אם יש)
    if kw_filtered and after_block:
        # מחפשים פוסט שלא עבר סינון מילות מפתח
        for p in after_block:
            if not passes_keyword_filter(p["content"]):
                snippet = p["content"].replace("\n", " ")[:150]
                lines.append(f"\n\U0001f4dd דוגמה לפוסט שנפסל (מילות מפתח):\n\"{snippet}...\"")
                break

    if blocked_count and after_dedup:
        for p in after_dedup:
            if is_blocked(p["content"]):
                snippet = p["content"].replace("\n", " ")[:150]
                # מציג את המילה החסומה שנמצאה
                with _kw_lock:
                    block_kws = _keywords_state["block"]
                found_word = ""
                text_lower = p["content"].lower()
                for kw in block_kws:
                    if kw in text_lower:
                        found_word = kw
                        break
                lines.append(f"\n\U0001f4dd דוגמה לפוסט שנחסם (מילה: '{found_word}'):\n\"{snippet}...\"")
                break

    return "\n".join(lines)


async def run_cycle():
    import gc
    # Lazy imports to keep startup light and allow running parts of the codebase
    # (like quiet-hours logic) without Playwright installed.
    from scraper import scrape_all
    from classifier import classify_batch
    from notifier import send_lead, send_error_alert, send_message

    _reset_scan_progress()

    log.info(f"===== סבב חדש: {_now_local().strftime('%Y-%m-%d %H:%M:%S %Z')} =====")

    # דיאגנוסטיקה — מוודאים שה-DB שורד בין סשנים
    from database import DB_PATH
    stats = get_stats()
    db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
    log.info(f"DB path: {DB_PATH} | size={db_size}B | seen={stats['seen']} sent={stats['sent']}")

    # טעינה מחדש של מילות מפתח וקבוצות מ-DB
    reload_keywords()
    try:
        from scraper import reload_groups
        reload_groups()
    except Exception:
        pass

    # ניקוי פוסטים ישנים (מעל 30 יום) מטבלת seen_posts
    try:
        deleted = cleanup_old_posts()
        if deleted:
            log.info(f"נמחקו {deleted} פוסטים ישנים מטבלת seen_posts")
    except Exception as e:
        log.warning(f"ניקוי פוסטים ישנים נכשל (ממשיכים בסריקה): {e}")

    # טעינת credentials מ-DB עם fallback למשתני סביבה
    fb_email = get_config("fb_email") or FB_EMAIL
    fb_password = get_config_encrypted("fb_password") or FB_PASSWORD

    if not fb_email or not fb_password:
        log.error("חסרים FB_EMAIL או FB_PASSWORD — לא ניתן לסרוק")
        _finish_scan_progress(0, error="חסרים פרטי חשבון")
        return

    # שלב 1 — גרידה (דפדפן פתוח)
    # מעדכנים total_groups מראש — כדי שהפרוגרס בר יראה את המצב כבר בזמן הקבוצה הראשונה
    try:
        from scraper import GROUPS
        scan_progress["total_groups"] = len(GROUPS)
    except Exception:
        pass
    try:
        posts = await scrape_all(fb_email, fb_password, seen_checker=is_seen,
                                 on_group_done=_on_group_scraped)
        log.info(f"סך הכל פוסטים שנמצאו: {len(posts)}")
    except Exception as e:
        log.error(f"שגיאה בגרידה: {e}", exc_info=True)
        _finish_scan_progress(0, error=str(e))
        try:
            await asyncio.to_thread(send_error_alert, f"שגיאה בגרידה: {e}")
        except Exception:
            pass
        return

    # הדפדפן כבר סגור כאן — משחררים זיכרון לפני הסיווג
    gc.collect()
    log.debug("זיכרון שוחרר אחרי סגירת דפדפן")

    # מעקב בריאות קבוצות — כמה פוסטים כל קבוצה החזירה
    try:
        from scraper import GROUPS
        _posts_by_group: dict[str, int] = {}
        for p in posts:
            g_url = p.get("group_url", "")
            if g_url:
                _posts_by_group[g_url] = _posts_by_group.get(g_url, 0) + 1
        for group in GROUPS:
            g_url = group["url"]
            count = _posts_by_group.get(g_url, 0)
            update_group_health(g_url, count)
        # בדיקת קבוצות לא פעילות — שליחת התראה
        try:
            _inactive_threshold = int(get_config("inactive_group_threshold", "50"))
        except (ValueError, TypeError):
            _inactive_threshold = 50
        if _inactive_threshold < 1:
            _inactive_threshold = 50
        for health in get_all_group_health():
            if health["consecutive_empty"] >= _inactive_threshold:
                # שולחים התראה רק בכפולות של הסף — לא להציף בהתראות חוזרות
                if health["consecutive_empty"] % _inactive_threshold == 0:
                    from notifier import send_message as _send_msg
                    _group_name = health["group_url"].split("/groups/")[-1].rstrip("/") or health["group_url"]
                    # חיפוש שם ידידותי מרשימת הקבוצות
                    for g in GROUPS:
                        if g["url"] == health["group_url"]:
                            _group_name = g["name"]
                            break
                    await asyncio.to_thread(
                        _send_msg,
                        f"\u26a0\ufe0f הקבוצה '{_group_name}' לא מחזירה פוסטים כבר {health['consecutive_empty']} סבבים — כדאי לבדוק",
                        disable_web_page_preview=True,
                    )
    except Exception as e:
        log.warning(f"מעקב בריאות קבוצות נכשל (ממשיכים): {e}")

    # try/finally — מבטיח ש-_finish_scan_progress ייקרא גם אם יש exception
    # אחרי שלב הגרידה (בסינון, סיווג, שליחה וכו')
    leads_found = 0
    try:
        # עדכון סטטוס — עוברים לשלב סינון
        scan_progress.update({"phase": "filtering", "phase_label": "מסנן פוסטים"})

        # שלב 2 — סינון כפילויות (בלי דפדפן — זיכרון נמוך)
        # לפוסטים עם URL אמיתי — בודקים לפי post_id ב-DB.
        # לפוסטים ללא URL אמיתי — ה-ID מבוסס hash תוכן (ייחודי לכל פוסט),
        # כך שגם הם נבדקים כראוי מול seen_posts.
        # בנוסף, בודקים content_hash מול sent_leads — תופס כפילויות כש-post_id
        # השתנה בין סשנים (URL extraction לא עקבי) אבל התוכן זהה.
        skipped_early_hash = 0
        new_posts = []
        for p in posts:
            if is_seen(p["id"]):
                continue
            c_hash = _content_dedup_hash(p["content"])
            p["_content_hash"] = c_hash  # שמירה — חוסך חישוב חוזר בשלבים הבאים
            if c_hash and is_content_hash_sent(c_hash):
                skipped_early_hash += 1
                snippet = p["content"].replace('\n', ' ')[:60]
                log.info(f"[DEDUP-EARLY] content_hash caught dup at seen-filter stage | id={p['id'][:20]} | hash={c_hash[:12]} | \"{snippet}...\"")
                mark_seen(p["id"], p["group"])
                continue
            new_posts.append(p)
        if skipped_early_hash:
            log.info(f"סינון כפילויות מוקדם: {skipped_early_hash} לפי content_hash (post_id השתנה בין סשנים)")

        # שלב 2.1 — איחוד פוסטים זהים מקבוצות שונות (cross-group dedup)
        # אותו אדם מפרסם נוסח זהה ב-3 קבוצות → ליד אחד עם ציון קבוצות נוספות.
        # חוסך קריאות AI מיותרות + מפחית רעש בטלגרם.
        _seen_hashes: dict[str, dict] = {}
        unique_posts = []
        cross_group_merged = 0
        for p in new_posts:
            c_hash = p.get("_content_hash", "")
            if not c_hash:
                # פוסט ללא hash (טקסט ריק/רעש בלבד) — מעביר כמו שהוא
                p["also_in"] = []
                p["_also_in_urls"] = []
                unique_posts.append(p)
                continue
            if c_hash in _seen_hashes:
                primary = _seen_hashes[c_hash]
                dup_group = p["group"]
                # רק אם באמת קבוצה אחרת — מונע תצוגה מטעה של "קבוצות נוספות"
                # כשיש שני פוסטים מאותה קבוצה עם hash זהה
                if dup_group != primary["group"] and dup_group not in primary["also_in"]:
                    primary["also_in"].append(dup_group)
                    # שומר group_url של הכפילות — כדי שבדיקת force_send תכסה
                    # גם מילות קבוצה ספציפיות מקבוצות שאוחדו
                    dup_url = p.get("group_url", "")
                    if dup_url:
                        primary["_also_in_urls"].append(dup_url)
                    cross_group_merged += 1
                # מסמן גם את הכפילות כנראית כדי שלא תופיע שוב בסבב הבא
                mark_seen(p["id"], p["group"])
            else:
                p["also_in"] = []
                p["_also_in_urls"] = []
                _seen_hashes[c_hash] = p
                unique_posts.append(p)
        if cross_group_merged:
            log.info(f"[DEDUP-XGROUP] איחוד פוסטים מקבוצות שונות: {cross_group_merged} כפילויות אוחדו")
        new_posts = unique_posts

        log.info(f"פוסטים חדשים (לא נראו קודם): {len(new_posts)}")
        scan_progress["new_posts"] = len(new_posts)

        # סימון כל הפוסטים כנראו (גם אלה שלא יעברו סינון מילות מפתח)
        for post in new_posts:
            mark_seen(post["id"], post["group"])

        # שלב 2.5 — סינון פוסטים ישנים (לפי max_post_age_days)
        # עוקף גם force_send — פוסט ישן לא יישלח גם אם מכיל מילת "שלח תמיד"
        max_age = _load_max_post_age()
        if max_age:
            before_age = len(new_posts)
            new_posts = [p for p in new_posts if not is_post_too_old(p["content"], max_age)]
            filtered_by_age = before_age - len(new_posts)
            if filtered_by_age:
                log.info(f"סונן לפי גיל: {filtered_by_age} פוסטים ישנים מ-{max_age} ימים דולגו")

        # שלב 2.6 — סינון מפרסמים חסומים (blocked_users)
        # לפני force_send — מפרסם חסום לא יישלח גם אם הפוסט מכיל מילת "שלח תמיד"
        with _kw_lock:
            _has_blocked_users = bool(_keywords_state["blocked_users"])
        if _has_blocked_users:
            before_user_block = len(new_posts)
            new_posts = [p for p in new_posts if not is_user_blocked(p.get("author_url", ""))]
            blocked_user_count = before_user_block - len(new_posts)
            if blocked_user_count:
                log.info(f"נחסמו {blocked_user_count} פוסטים לפי מפרסמים חסומים (blocked_users)")

        # שלב 3 — "שלח תמיד": פוסטים עם מילת force_send עוקפים את כל הסינון (block/pre_filter/AI)
        force_send_posts = []
        rest_posts = []
        for post in new_posts:
            # בדיקת force_send — גם מול group_url של הפוסט המקורי
            # וגם מול group_urls של קבוצות שאוחדו (cross-group dedup)
            matched_kw = matches_force_send(post["content"], post.get("group_url", ""))
            if not matched_kw:
                for alt_url in post.get("_also_in_urls", []):
                    matched_kw = matches_force_send(post["content"], alt_url)
                    if matched_kw:
                        break
            if matched_kw and not is_lead_sent(post["id"]):
                c_hash = post.get("_content_hash") or _content_dedup_hash(post["content"])
                if is_content_hash_sent(c_hash):
                    log.debug(f"דילוג על force_send — תוכן זהה כבר נשלח (content_hash)")
                    continue
                reason = f"שלח תמיד — מילת מפתח: {matched_kw}"
                hot_kw = matches_hot_word(post["content"])
                sent_ok = await asyncio.to_thread(
                    send_lead,
                    group_name=post["group"],
                    content=post["content"],
                    post_url=post["url"],
                    reason=reason,
                    has_real_url=post.get("has_real_url", True),
                    also_in=post.get("also_in"),
                    is_hot=bool(hot_kw),
                    author_url=post.get("author_url", ""),
                )
                if not sent_ok:
                    log.warning(f"שליחת force_send נכשלה — הליד לא יישמר ב-DB (id={post['id'][:20]})")
                    rest_posts.append(post)
                    continue
                save_lead(post["id"], post["group"], post["content"], reason,
                          content_hash=c_hash)
                if post.get("group_url"):
                    update_group_last_lead(post["group_url"])
                leads_found += 1
                scan_progress["leads_sent"] = leads_found
                _dedup_debug(post["id"], post["content"], c_hash, "force_send")
                force_send_posts.append(post)
            else:
                rest_posts.append(post)

        if force_send_posts:
            log.info(f"נשלחו {len(force_send_posts)} לידים ישירות (force_send)")

        # שלב 4א — סינון מילים חוסמות (BLOCK_KEYWORDS)
        with _kw_lock:
            _has_block = bool(_keywords_state["block"])
        if _has_block:
            before_block = len(rest_posts)
            rest_posts = [p for p in rest_posts if not is_blocked(p["content"])]
            blocked_count = before_block - len(rest_posts)
            if blocked_count:
                log.info(f"נחסמו {blocked_count} פוסטים לפי מילים חוסמות (BLOCK_KEYWORDS)")

        # שלב 4ב — סינון מילות מפתח לפני שליחה ל-AI
        filtered_posts = [p for p in rest_posts if passes_keyword_filter(p["content"])]
        skipped_by_filter = len(rest_posts) - len(filtered_posts)
        if skipped_by_filter:
            log.info(f"סונן לפי מילות מפתח: {skipped_by_filter} פוסטים לא רלוונטיים דולגו")
        log.info(f"פוסטים שעוברים לסיווג AI: {len(filtered_posts)}")

        if not filtered_posts:
            log.info("אין פוסטים רלוונטיים — מדלג על סיווג")

        # סינון פוסטים שכבר נשלחו כלידים — בדיקה לפי post_id + content_hash
        not_yet_sent = []
        skipped_by_id = 0
        skipped_by_hash = 0
        for p in filtered_posts:
            if is_lead_sent(p["id"]):
                skipped_by_id += 1
                continue
            c_hash = p.get("_content_hash") or _content_dedup_hash(p["content"])
            if is_content_hash_sent(c_hash):
                skipped_by_hash += 1
                snippet = p["content"].replace('\n', ' ')[:60]
                log.info(f"[DEDUP-CATCH] content_hash caught dup | id={p['id'][:20]} | hash={c_hash[:12]} | \"{snippet}...\"")
                continue
            p["_content_hash"] = c_hash
            not_yet_sent.append(p)
        if skipped_by_id or skipped_by_hash:
            log.info(f"סינון כפילויות: {skipped_by_id} לפי post_id, {skipped_by_hash} לפי content_hash")

        to_classify = not_yet_sent
        scan_progress["posts_to_classify"] = len(to_classify)

        if to_classify:
            # עדכון סטטוס — עוברים לשלב סיווג AI
            scan_progress.update({"phase": "classifying", "phase_label": "מסווג עם AI"})

            # סיווג באצ'ים — שליחת מספר פוסטים ל-AI בבקשה אחת
            # עוטף ב-asyncio.to_thread כי classify_batch משתמש ב-openai סינכרוני
            results = await asyncio.to_thread(classify_batch, to_classify)

            for post, result in zip(to_classify, results):
                if result.get("relevant"):
                    reason = result.get("reason", "")
                    hot_kw = matches_hot_word(post["content"])
                    sent_ok = await asyncio.to_thread(
                        send_lead,
                        group_name=post["group"],
                        content=post["content"],
                        post_url=post["url"],
                        reason=reason,
                        has_real_url=post.get("has_real_url", True),
                        also_in=post.get("also_in"),
                        is_hot=bool(hot_kw),
                        author_url=post.get("author_url", ""),
                    )
                    if not sent_ok:
                        log.warning(f"שליחת ליד AI נכשלה — הליד לא יישמר ב-DB (id={post['id'][:20]})")
                        continue
                    save_lead(post["id"], post["group"], post["content"], reason,
                              content_hash=post.get("_content_hash", ""))
                    if post.get("group_url"):
                        update_group_last_lead(post["group_url"])
                    leads_found += 1
                    scan_progress["leads_sent"] = leads_found
                    _dedup_debug(post["id"], post["content"], post.get("_content_hash", ""), "ai_classified")
                else:
                    log.debug(f"לא רלוונטי: {result.get('reason', '')}")

        _finish_scan_progress(leads_found)
    except Exception as e:
        log.error(f"שגיאה לאחר גרידה: {e}", exc_info=True)
        _finish_scan_progress(leads_found, error=str(e))
        raise

    stats = get_stats()
    log.info(f"סבב הסתיים | לידים בסבב: {leads_found} | סך נראו: {stats['seen']} | סך נשלחו: {stats['sent']}")

    # הודעת סיכום סריקה — נשלחת לטלגרם רק כשנמצאו לידים חדשים,
    # כדי לעזור למשתמש להבחין בין הודעות מסריקות שונות.
    if leads_found > 0:
        scan_time = _now_local().strftime("%H:%M")
        summary = f"עד כאן לסריקה של שעה {scan_time}, נתראה בסריקה הבאה 💫"
        try:
            await asyncio.to_thread(send_message, summary, disable_web_page_preview=True)
        except Exception:
            log.debug("שליחת הודעת סיכום סריקה נכשלה — לא קריטי")

async def main():
    log.info("מאתחל מסד נתונים...")
    init_db()
    # פאנל מופעל ראשון — אם הוא תופס את אותו פורט כמו health, לא צריך שרת health נפרד
    panel_started = _start_panel()
    panel_port = _get_panel_port()
    if panel_started and HEALTH_PORT == panel_port:
        log.info(f"הפאנל רץ על פורט {HEALTH_PORT} — שרת health נפרד לא נדרש")
    else:
        start_health_server()
    with _kw_lock:
        _startup_block = list(_keywords_state['block'])
    log.info(
        f"מתחיל. סבב כל {INTERVAL_MINUTES} דקות. "
        f"LOG_LEVEL={os.environ.get('LOG_LEVEL', 'DEBUG')} "
        f"TIMEZONE={TIMEZONE_NAME} QUIET_HOURS={QUIET_HOURS or '(disabled)'} "
        f"BLOCK_KEYWORDS={', '.join(_startup_block) if _startup_block else '(disabled)'}"
    )

    # shared_state — מילון משותף בין הלולאה הראשית לטלגרם.
    # כולל quiet hours עדכני, מצב סריקה, וכו'.
    shared_state: dict = {
        "quiet": _load_quiet_hours_from_db(),
        "scan_in_progress": False,
        "last_scan_started": None,
        "last_scan_finished": None,
    }
    # מעקב זמן אוטומציה יומי (בשניות) — מודד זמן סריקה בפועל בלבד
    daily_automation: dict = {
        "date": _now_local().date(),
        "total_seconds": 0.0,
    }
    # חשיפת shared_state ל-health check endpoint
    global _health_shared_state
    _health_shared_state = shared_state

    scan_now_event = asyncio.Event()
    scan_force_event = asyncio.Event()
    scan_lock = asyncio.Lock()

    def _check_daily_limit() -> bool:
        """בודק אם חרגנו ממגבלת הזמן היומית. מחזיר True אם מותר לסרוק."""
        if DAILY_AUTOMATION_LIMIT_MINUTES <= 0:
            return True
        today = _now_local().date()
        if daily_automation["date"] != today:
            daily_automation["date"] = today
            daily_automation["total_seconds"] = 0.0
            shared_state["daily_limit_notified"] = False
        limit_secs = DAILY_AUTOMATION_LIMIT_MINUTES * 60
        if daily_automation["total_seconds"] >= limit_secs:
            return False
        return True

    async def run_scan_guarded(force: bool = False) -> bool:
        """מריץ סריקה. מחזיר True אם הסריקה רצה, False אם דולגה בגלל מגבלה יומית."""
        async with scan_lock:
            if not force and not _check_daily_limit():
                used = int(daily_automation["total_seconds"] / 60)
                log.info(
                    f"מגבלת אוטומציה יומית הושגה ({used}/{DAILY_AUTOMATION_LIMIT_MINUTES} דקות) — מדלג"
                )
                return False
            shared_state["scan_in_progress"] = True
            shared_state["last_scan_started"] = _now_local()
            scan_start = asyncio.get_event_loop().time()
            try:
                await run_cycle()
            finally:
                elapsed = asyncio.get_event_loop().time() - scan_start
                today = _now_local().date()
                if daily_automation["date"] != today:
                    daily_automation["date"] = today
                    daily_automation["total_seconds"] = 0.0
                    shared_state["daily_limit_notified"] = False
                daily_automation["total_seconds"] += elapsed
                shared_state["scan_in_progress"] = False
                shared_state["last_scan_finished"] = _now_local()
                used = int(daily_automation["total_seconds"] / 60)
                log.debug(f"זמן אוטומציה יומי: {used} דקות מתוך {DAILY_AUTOMATION_LIMIT_MINUTES or '∞'}")
        return True

    # שמירת רפרנס ל-task כדי למנוע garbage collection
    _control_task = None
    if TELEGRAM_CONTROL:
        _control_task = asyncio.create_task(
            _telegram_control_loop(
                scan_now_event=scan_now_event,
                scan_force_event=scan_force_event,
                shared_state=shared_state,
            )
        )

    while True:
        # טעינה מחדש של הגדרות מהפאנל (interval, quiet hours)
        interval = _load_interval_from_db()
        quiet = _load_quiet_hours_from_db()
        shared_state["quiet"] = quiet  # עדכון כדי שגם לולאת טלגרם תראה את הערך החדש

        try:
            did_scan = False
            if scan_now_event.is_set():
                force = scan_force_event.is_set()
                scan_now_event.clear()
                scan_force_event.clear()

                if not force and quiet and _is_quiet_now(_now_local(), quiet):
                    # Should not happen (we don't schedule /scan during quiet hours),
                    # but guard anyway. did_scan stays False → quiet-hours sleep path runs.
                    log.info("בקשת סריקה נדחתה בגלל שעות שקטות")
                else:
                    did_scan = True  # פקודה טופלה — לא לנסות סריקה מתוזמנת
                    log.info(f"מריץ סריקה לפי פקודה (force={force})")
                    ran = await run_scan_guarded(force=force)
                    if not ran:
                        # מגבלת אוטומציה יומית — עדכון המשתמש בטלגרם
                        used = int(daily_automation["total_seconds"] / 60)
                        try:
                            from notifier import send_message as _notify
                            await asyncio.to_thread(
                                _notify,
                                f"⏳ מגבלת אוטומציה יומית הושגה ({used}/{DAILY_AUTOMATION_LIMIT_MINUTES} דקות). השתמש /scan_force לסריקה בכל זאת.",
                            )
                        except Exception:
                            pass

            if not did_scan:
                now = _now_local()
                qh_db = get_config("quiet_hours", _CONFIG_NOT_SET)
                quiet_spec = qh_db if qh_db is not _CONFIG_NOT_SET else QUIET_HOURS
                if quiet and _is_quiet_now(now, quiet):
                    secs = _seconds_until_quiet_end(now, quiet)
                    sleep_for = max(60, min(int(secs) + 1, 12 * 60 * 60))
                    log.info(
                        f"שעות שקטות פעילות ({quiet_spec}, {TIMEZONE_NAME}) — "
                        f"מדלג על סריקה. ישן {sleep_for//60} דקות (או /scan_force)."
                    )
                    try:
                        await asyncio.wait_for(scan_now_event.wait(), timeout=sleep_for)
                    except asyncio.TimeoutError:
                        pass
                    continue

                ran = await run_scan_guarded()
                if not ran and not shared_state.get("daily_limit_notified"):
                    # הודעה חד-פעמית ליום שהמגבלה הושגה בסריקות אוטומטיות
                    shared_state["daily_limit_notified"] = True
                    used = int(daily_automation["total_seconds"] / 60)
                    try:
                        from notifier import send_message as _notify
                        await asyncio.to_thread(
                            _notify,
                            f"⏳ מגבלת אוטומציה יומית הושגה ({used}/{DAILY_AUTOMATION_LIMIT_MINUTES} דקות). סריקות אוטומטיות מופסקות להיום.",
                        )
                    except Exception:
                        pass
        except Exception as e:
            log.error(f"שגיאה לא צפויה: {e}", exc_info=True)
            try:
                from notifier import send_error_alert
                await asyncio.to_thread(send_error_alert, f"שגיאה לא צפויה: {e}")
            except Exception:
                pass

        log.info(f"ממתין {interval} דקות לסבב הבא... (או /scan)")
        try:
            await asyncio.wait_for(scan_now_event.wait(), timeout=interval * 60)
        except asyncio.TimeoutError:
            pass

if __name__ == "__main__":
    asyncio.run(main())
