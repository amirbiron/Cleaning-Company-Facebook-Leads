"""
סורק מכרזים ממשלתיים מ-data.gov.il (CKAN API).
מחפש מכרזי ניקיון ותחזוקה פעילים ושולח לידים לטלגרם.

משתני סביבה:
  TENDER_KEYWORDS — מילות חיפוש (מופרד בפסיקים). ברירת מחדל: ניקיון,תחזוקה,אחזקה,ניקוי
  TENDER_ENABLED  — "1" להפעלה (ברירת מחדל: "0")
"""

import hashlib
import os
import re
from datetime import datetime
from urllib.parse import quote

import requests

from logger import get_logger

log = get_logger("Tenders")

# ── הגדרות ──────────────────────────────────────────────────
# מילות חיפוש — ניתן להגדיר דרך משתנה סביבה או דרך DB (פאנל/טלגרם)
_TENDER_KEYWORDS_ENV = os.environ.get(
    "TENDER_KEYWORDS", "ניקיון,תחזוקה,אחזקה,ניקוי"
).strip()
TENDER_KEYWORDS_DEFAULT = [
    w.strip() for w in _TENDER_KEYWORDS_ENV.split(",") if w.strip()
]

TENDER_ENABLED = os.environ.get("TENDER_ENABLED", "0").strip().lower() in (
    "1", "true", "yes", "on",
)

# ── CKAN API ────────────────────────────────────────────────
_BASE_URL = "https://data.gov.il/api/3/action"
_DATASET_ID = "tenders"

# קאש של resource_id — נטען פעם אחת
_resource_id: str | None = None


def _get_resource_id() -> str | None:
    """מחזיר את ה-resource_id הראשון של מאגר המכרזים ב-data.gov.il."""
    global _resource_id
    if _resource_id:
        return _resource_id
    try:
        resp = requests.get(
            f"{_BASE_URL}/package_show",
            params={"id": _DATASET_ID},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        resources = data.get("result", {}).get("resources", [])
        if not resources:
            log.error("לא נמצאו resources במאגר המכרזים")
            return None
        _resource_id = resources[0]["id"]
        log.info(f"resource_id למכרזים: {_resource_id}")
        return _resource_id
    except Exception as e:
        log.error(f"שגיאה בטעינת resource_id: {e}")
        return None


def _search_tenders(keyword: str, limit: int = 100) -> list[dict]:
    """מחפש מכרזים לפי מילת מפתח דרך CKAN datastore_search."""
    rid = _get_resource_id()
    if not rid:
        return []
    try:
        resp = requests.get(
            f"{_BASE_URL}/datastore_search",
            params={
                "resource_id": rid,
                "q": keyword,
                "limit": limit,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        records = data.get("result", {}).get("records", [])
        return records
    except Exception as e:
        log.error(f"שגיאה בחיפוש מכרזים (keyword={keyword}): {e}")
        return []


def _tender_id(record: dict) -> str:
    """מחלץ מזהה ייחודי למכרז.
    משתמש ב-מספר פרסום (_id או publication_id) אם קיים,
    אחרת hash על השם + גוף מפרסם.
    """
    # CKAN datastore כולל _id אוטומטי
    if record.get("_id"):
        return f"tender_{record['_id']}"
    # ניסיון לפי מספר מכרז
    for field in ("מספר מכרז", "מספר פרסום", "tender_number", "publication_number"):
        val = record.get(field, "")
        if val:
            return f"tender_{val}"
    # fallback — hash על תוכן
    raw = f"{record.get('שם המכרז', '')}{record.get('גורם מפרסם', '')}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"tender_hash_{h}"


def _parse_date(raw: str) -> datetime | None:
    """מנסה לפרסר תאריך ממספר פורמטים אפשריים."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    return None


def _is_active(record: dict) -> bool:
    """בודק אם מכרז עדיין פעיל (מועד הגשה עתידי או לא מוגדר)."""
    now = datetime.now()
    # מנסים שדות שונים של תאריך הגשה
    for field in ("תאריך אחרון להגשה", "מועד אחרון להגשה", "submission_date", "last_date"):
        raw = record.get(field, "")
        if raw:
            dt = _parse_date(str(raw))
            if dt:
                return dt > now
    # אם אין תאריך הגשה — מניחים שפעיל
    return True


def _build_mr_gov_url(record: dict) -> str:
    """בונה לינק לעמוד המכרז ב-mr.gov.il."""
    for field in ("מספר פרסום", "publication_number", "_id"):
        val = record.get(field, "")
        if val:
            return f"https://mr.gov.il/ilgstorefront/he/p/{val}"
    return "https://mr.gov.il/ilgstorefront/he/search/?s=TENDER"


def _format_tender(record: dict) -> dict:
    """ממיר רשומת CKAN למבנה ליד מנורמל."""
    # מיפוי שדות — data.gov.il משתמש בשמות עבריים
    name = (
        record.get("שם המכרז", "")
        or record.get("tender_name", "")
        or record.get("שם", "")
        or "מכרז ללא שם"
    )
    publisher = (
        record.get("גורם מפרסם", "")
        or record.get("publisher", "")
        or record.get("שם המפרסם", "")
        or ""
    )
    deadline_raw = (
        record.get("תאריך אחרון להגשה", "")
        or record.get("מועד אחרון להגשה", "")
        or record.get("submission_date", "")
        or ""
    )
    deadline = ""
    if deadline_raw:
        dt = _parse_date(str(deadline_raw))
        if dt:
            deadline = dt.strftime("%d/%m/%Y %H:%M")
        else:
            deadline = str(deadline_raw)

    category = record.get("תחום", "") or record.get("category", "") or ""
    url = _build_mr_gov_url(record)
    tid = _tender_id(record)

    return {
        "id": tid,
        "name": name,
        "publisher": publisher,
        "deadline": deadline,
        "category": category,
        "url": url,
        "source": "mr.gov.il",
    }


def fetch_tenders(keywords: list[str] | None = None) -> list[dict]:
    """מחפש מכרזים לפי מילות מפתח ומחזיר רשימת מכרזים מפורמטים.

    keywords — רשימת מילות חיפוש. אם None, משתמש ב-TENDER_KEYWORDS_DEFAULT.
    """
    if keywords is None:
        keywords = TENDER_KEYWORDS_DEFAULT

    if not keywords:
        log.warning("אין מילות חיפוש למכרזים — מדלג")
        return []

    log.info(f"מחפש מכרזים עם מילות מפתח: {', '.join(keywords)}")

    all_tenders: dict[str, dict] = {}  # dedup לפי id

    for kw in keywords:
        records = _search_tenders(kw)
        log.debug(f"  '{kw}' — {len(records)} תוצאות")
        for record in records:
            if not _is_active(record):
                continue
            formatted = _format_tender(record)
            if formatted["id"] not in all_tenders:
                all_tenders[formatted["id"]] = formatted

    result = list(all_tenders.values())
    log.info(f"סה\"כ מכרזים פעילים שנמצאו: {len(result)}")
    return result


def load_tender_keywords() -> list[str]:
    """טוען מילות חיפוש מכרזים מ-DB עם fallback למשתנה סביבה."""
    try:
        from database import get_config
        raw = get_config("tender_keywords")
        if raw:
            import json
            words = json.loads(raw)
            if isinstance(words, list) and words:
                return words
    except Exception:
        pass
    return TENDER_KEYWORDS_DEFAULT


def save_tender_keywords(keywords: list[str]):
    """שומר מילות חיפוש מכרזים ב-DB."""
    import json
    from database import set_config
    set_config("tender_keywords", json.dumps(keywords))
