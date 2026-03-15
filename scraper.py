import asyncio
import functools
import hashlib
import html as html_mod
import json
import os
import random
import re
from pathlib import Path

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from logger import get_logger

log = get_logger("Scraper")

SESSION_FILE = Path(__file__).resolve().parent / "data" / "fb_session.json"

# כותרות גנריות/שגויות שלא מייצגות שם קבוצה אמיתי
_BAD_TITLES = {"facebook", "content not found", "page not found",
               "error", "log in", "התחברות", "תוכן לא נמצא", "שגיאה"}

# טקסטים של overlay לוגין במובייל — הצירוף הזה ייחודי לאוברליי ולא מופיע בפוסטים
_LOGIN_OVERLAY_MARKERS = ("פתיחת האפליקציה", "התחברות")

# סלקטורים לכפתור login/submit במובייל — משותף ללוגין רגיל ולוגין דו-שלבי.
# ~= (contains-word) במקום = כי data-sigil משתמש ב-tokens מופרדי רווחים.
_LOGIN_BTN_SELECTORS = [
    "button[name='login']",
    "input[name='login']",
    "button[data-sigil~='m_login_button']",
    "[data-sigil~='m_login_button']",
    "button[type='submit']",
    "input[type='submit']",
]

# סימני checkpoint / אימות נוסף בURL אחרי לוגין
_CHECKPOINT_MARKERS = ("checkpoint", "two_step_verification", "approvals")

# User-Agent דסקטופ ללוגין — www.facebook.com מציג form HTML אמיתי עם UA דסקטופ
_DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ניווט עם retry — timeout רשת חד-פעמי לא צריך להרוג את כל הסייקל
_GOTO_RETRIES = 3
_GOTO_TIMEOUTS = [30_000, 45_000, 60_000]  # timeout עולה בכל ניסיון


async def _goto_with_retry(page, url: str, **kwargs):
    """page.goto עם retry על timeout — מונע קריסת סייקל בגלל איטיות רשת חולפת."""
    kwargs.pop("timeout", None)  # timeout מנוהל פנימית — מונע duplicate kwarg
    last_err = None
    for attempt in range(_GOTO_RETRIES):
        timeout = _GOTO_TIMEOUTS[min(attempt, len(_GOTO_TIMEOUTS) - 1)]
        try:
            return await page.goto(url, timeout=timeout, **kwargs)
        except PlaywrightTimeout as e:
            last_err = e
            log.warning(f"timeout בניווט ל-{url[:60]} (ניסיון {attempt + 1}/{_GOTO_RETRIES}, "
                        f"timeout={timeout}ms) — {'מנסה שוב' if attempt + 1 < _GOTO_RETRIES else 'נכשל'}")
            if attempt + 1 < _GOTO_RETRIES:
                await asyncio.sleep(2 * (attempt + 1))
    raise last_err


async def _has_login_overlay(page) -> bool:
    """מזהה overlay לוגין לפי טקסט — כשפייסבוק מציג 'פתיחת האפליקציה / התחברות'
    בלי redirect, כך שבדיקת URL לא מספיקה.

    הבחנה מבאנר UI רגיל: overlay חוסם = דף כמעט ריק (< 500 תווים),
    באנר רגיל = דף מלא בתוכן קבוצה (הרבה יותר מ-500 תווים)."""
    try:
        result = await page.evaluate("""() => {
            const text = document.body?.innerText || '';
            return { top: text.substring(0, 500), len: text.length };
        }""")
        top_text = result["top"]
        total_len = result["len"]
        if not all(m in top_text for m in _LOGIN_OVERLAY_MARKERS):
            return False
        # דף עם תוכן אמיתי (קבוצה + באנר UI) — לא overlay חוסם
        if total_len > 500:
            return False
        return True
    except Exception:
        return False


async def _is_wui_page(page) -> bool:
    """בודק אם הדף מציג רק לינקי wui/action ('פתח באפליקציה') — סימן לסשן פג.

    פייסבוק מובייל לפעמים לא עושה redirect ללוגין כשהסשן פג,
    אלא מציג את התוכן עם לינקי wui בלבד (הורד אפליקציה / פתח בכרום וכו').
    במצב הזה אין אף <a href> עם קישור לפוסט.

    סף מחמיר: לפחות 3 לינקי wui שמהווים לפחות 75% מכלל הלינקים.
    סף נמוך יותר גורם ל-false positive כי פייסבוק מוסיף כמה לינקי wui
    גם בדפים תקינים (עטיפת מובייל, באנר 'הורד אפליקציה')."""
    try:
        result = await page.evaluate("""() => {
            const links = [...document.querySelectorAll('a[href]')];
            if (links.length < 3) return false;
            const wui = links.filter(a =>
                (a.getAttribute('href') || '').includes('/wui/'));
            return wui.length >= 3 && wui.length >= links.length * 0.75;
        }""")
        return result
    except Exception:
        return False


# ביטויים דינאמיים שמשתנים בין סריקות — מוסרים לפני חישוב hash.
# טיימסטמפים דינמיים — פייסבוק מציג "8 שעות" ומעדכן לאורך זמן.
# פטרנים inline (לא מעוגנים) מסירים "8 שעות" גם מתוך שורת מטא-דאטה כמו
# "אמיר בירון 8 שעות [אייקונים] היי חברים". סיכון נמוך: עלול להסיר "5 שעות"
# מתוך "עבודה של 5 שעות ביום", אבל hash יציב עדיף על כפילויות חוזרות בכל סריקה.
_TIMESTAMP_RE = re.compile(
    # --- מעוגנים (שורה שלמה בלבד) ---
    # פורמט קצר אנגלי: "3h", "2d" — מעוגן כי "3d" עלול להתנגש עם תוכן
    r'^\d+\s*[hdmwHDMW]$'
    r'|^עכשיו$|^אתמול$|^היום$|^just now$|^yesterday$|^today$'
    # --- לא-מעוגנים (inline) ---
    # "לפני X שעות" — prefix ייחודי ל-timestamps
    r'|לפני[ \t]+\d+[ \t]*(?:שע|דק|יו|ימ|שני|חוד|שבוע)\w*'
    # "אתמול ב-15:30" / "היום ב-10:00"
    r'|(?:אתמול|היום)[ \t]+ב-?[ \t]*\d{1,2}:\d{2}'
    # זמנים יחסיים בעברית: "8 שעות", "3 דקות", "2 ימים"
    # [ \t]* במקום \s* — מונע התאמה בין שורות (50\nשעות לא יימחק)
    r'|\d+[ \t]*(?:שעות|שעה|דקות|דקה|ימים|יום|שבועות|שבוע|חודשים|חודש|שניות|שנייה|שנים|שנה)\b'
    # זמנים יחסיים באנגלית: "5 hours", "2 days", "30 mins"
    r'|\d+[ \t]*(?:hours?|minutes?|seconds?|days?|weeks?|months?|hrs?|mins?|secs?|wks?)\b'
    # --- תאריכים עבריים: "4 במרץ", "15 בינואר בשעה 10:30" ---
    # פייסבוק עובר מזמן יחסי ("8 שעות") לתאריך מלא אחרי ~24 שעות
    r'|\d{1,2}(?!\d)[ \t]+ב(?:ינואר|פברואר|מרץ|מרס|אפריל|מאי|יוני|יולי|אוגוסט|ספטמבר|אוקטובר|נובמבר|דצמבר)'
    r'(?:[ \t]+בשעה[ \t]+\d{1,2}:\d{2})?'
    # --- תאריכים אנגליים: "March 4", "Jan 15 at 10:30 AM" ---
    # (?!\d) מונע התאמה חלקית למספרי שנה: "March 2024" לא יתפס כ-"March 20"
    r'|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[ \t]+\d{1,2}(?!\d)'
    r'(?:[ \t]+at[ \t]+\d{1,2}:\d{2}(?:[ \t]*[ap]m)?)?',
    re.MULTILINE | re.IGNORECASE
)

# שורות engagement שמשתנות בין סריקות (מספר תגובות/לייקים/שיתופים).
# מעוגן לשורה שלמה כדי לא למחוק "5 תגובות" מתוך "קיבלתי 5 תגובות על הפוסט".
# תומך בפורמט בודד ("5 תגובות") ומשולב ("5 תגובות · 3 שיתופים").
_ENGAGEMENT_WORDS_PATTERN = (
    r'(?:תגוב\w*|שיתו[פף]\w*|לייק\w*|צפי\w*'
    r'|comments?|shares?|likes?|reactions?|views?)'
)
_ENGAGEMENT_LINE_RE = re.compile(
    # תומך גם בשורות שמתחילות באמוג'י ריאקציה (👍❤️) לפני המספר
    r'^(?:[^\w\s]*\s*)?\d[\d,.]*\s*' + _ENGAGEMENT_WORDS_PATTERN
    + r'(?:\s*[·•|,]\s*\d[\d,.]*\s*' + _ENGAGEMENT_WORDS_PATTERN + r')*\s*$',
    re.IGNORECASE
)


# מילות רעש UI של פייסבוק — מופיעות/נעלמות בין סריקות.
# lower-case כי הטקסט מנורמל ל-lowercase לפני הבדיקה.
_NOISE_WORDS = frozenset({
    'like', 'comment', 'share', 'send message', 'see more', 'see less',
    'see translation', 'write a comment...', 'write a comment…',
    'suggested for you', 'sponsored', 'all comments',
    'most relevant', 'newest',
    'הגב', 'שתף', 'שתפ', 'אהבתי', 'קרא עוד', 'ראה עוד', 'הצג עוד',
    'ראה תרגום', 'כתוב תגובה...', 'כתוב תגובה…', 'כתוב תגובה',
    'שלח הודעה', 'מומלץ', 'מומלצת', 'מומלצים', 'ממומן',
    'כל התגובות', 'הגב/י', 'שתף/י',
    'עוד', '... עוד', '… עוד', 'הרלוונטיות ביותר', 'החדשות ביותר',
})

# שורות דינמיות נוספות שמשתנות בין סריקות ולא נתפסות ע"י _ENGAGEMENT_LINE_RE.
# "+N" — ספירת ריאקשנים/תגובות ("ו-עוד 3" / "+3").
# "ו-עוד N אחרים" — רשימת מגיבים.
# מעוגנים לשורה שלמה כדי לא למחוק "+3" מתוך מספר טלפון.
_DYNAMIC_LINE_RE = re.compile(
    r'^\+\d+$'
    r'|^ו-עוד\s+\d+\s*(?:אחרי[םם]?)?$'
    r'|^and\s+\d+\s+(?:more|others?)$'
    # "פלוני ו-3 אחרים" / "פלוני, אלמוני ו-3 אחרים" — רשימת מגיבים/ריאקציות
    # שם = עד 2 מילים ללא פסיק. רשימת שמות = שמות מופרדים בפסיקים.
    # מגביל ל-2 מילים כדי לא לתפוס משפטי תוכן אמיתיים.
    r'|^\S+(?:\s+\S+)?\s+ו-?\s*\d+\s*(?:אחרי[םם]?|נוספי[םם]?)\s*$'
    r'|^\S+(?:\s+\S+)?,(?:\s*\S+(?:\s+\S+)?,)*\s*\S+(?:\s+\S+)?\s+ו-?\s*\d+\s*(?:אחרי[םם]?|נוספי[םם]?)\s*$'
    # "John and 3 others" / "John, Jane and 3 others" — English variant
    r'|^\S+(?:\s+\S+)?\s+and\s+\d+\s+(?:more|others?)\s*$'
    r'|^\S+(?:\s+\S+)?,(?:\s*\S+(?:\s+\S+)?,)*\s*\S+(?:\s+\S+)?\s+and\s+\d+\s+(?:more|others?)\s*$',
    re.IGNORECASE
)

# שורות עם אמוג'י ללא אותיות — כמו "👍 5", "👍❤️ 12", "😂🔥".
# שורות כאלה הן ריאקציות/ספירות דינמיות שמשתנות בין סריקות.
# דורשים נוכחות אמוג'י כדי לא לתפוס מספרי טלפון (054-1234567).
_EMOJI_RE = re.compile(
    r'[\U0001F600-\U0001F64F'   # Emoticons
    r'\U0001F300-\U0001F5FF'    # Misc Symbols and Pictographs
    r'\U0001F680-\U0001F6FF'    # Transport and Map
    r'\U0001F1E0-\U0001F1FF'    # Flags
    r'\U0001F900-\U0001F9FF'    # Supplemental Symbols
    r'\U0001FA00-\U0001FA6F'    # Chess Symbols
    r'\U0001FA70-\U0001FAFF'    # Symbols Extended-A
    r'\u2600-\u27BF'            # Misc Symbols
    r'\u2702-\u27B0'            # Dingbats
    r'\uFE0F]'                  # Variation Selector
)

# בר פעולות משולב — "אהבתי · תגובה · שיתוף" וכו'.
# מופיע כשורה שלמה עם מילות UI מופרדות ב-· או |.
_ACTION_BAR_WORDS = frozenset({
    'like', 'comment', 'share', 'send', 'reply',
    'אהבתי', 'תגובה', 'שיתוף', 'שיתו', 'שתף', 'שתפ',
    'הגב', 'שלח', 'שלח הודעה', 'הגב/י', 'שתף/י',
})
def _is_action_bar(line: str) -> bool:
    """בודק אם שורה היא בר פעולות UI (כמו 'אהבתי · תגובה · שיתוף')."""
    parts = re.split(r'\s*[·•|]\s*', line)
    if len(parts) < 2:
        return False
    non_empty = [p.strip() for p in parts if p.strip()]
    if not non_empty:
        return False
    return all(p in _ACTION_BAR_WORDS for p in non_empty)

# רגקס להסרת URLs (עם פרמטרי מעקב דינמיים כמו fbclid שמשתנים בין טעינות)
_URL_RE = re.compile(r'https?://\S+', re.IGNORECASE)

# תווי Unicode בלתי-נראים שפייסבוק מוסיף — כיווניות, zero-width, soft-hyphen
_INVISIBLE_CHARS_RE = re.compile(
    r'[\u200e\u200f\u200b\u200c\u200d\u2060\u2066\u2067\u2068\u2069\ufeff\xad]'
)

# Private Use Area — אייקונים מותאמים של פייסבוק שמשתנים בין רינדורים.
# כולל BMP PUA (E000-F8FF) + Supplementary PUA-A/B (F0000+, 100000+).
# אייקוני פרטיות/גלובוס של פייסבוק (כמו 󰞋󱟠 󳄫) נמצאים ב-supplementary planes.
_PUA_RE = re.compile(r'[\uE000-\uF8FF\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]')


# רגקס משולב לבדיקת שורות שצריך לדלג עליהן — מאחד digits-only + engagement + dynamic
# במקום 3 בדיקות regex נפרדות לכל שורה
_SKIP_LINE_RE = re.compile(
    # שורות שהן רק מספרים וסימני פיסוק (לייקים, ספירות)
    r'^\d[\d,. ]*$'
    # שורות engagement ("5 תגובות", "👍 5 תגובות · 3 שיתופים")
    r'|^(?:[^\w\s]*\s*)?\d[\d,.]*\s*' + _ENGAGEMENT_WORDS_PATTERN
    + r'(?:\s*[·•|,]\s*\d[\d,.]*\s*' + _ENGAGEMENT_WORDS_PATTERN + r')*\s*$'
    # שורות דינמיות ("+3", "ו-עוד 5 אחרים", "פלוני ו-3 אחרים")
    r'|^\+\d+$'
    r'|^ו-עוד\s+\d+\s*(?:אחרי[םם]?)?$'
    r'|^and\s+\d+\s+(?:more|others?)$'
    r'|^\S+(?:\s+\S+)?\s+ו-?\s*\d+\s*(?:אחרי[םם]?|נוספי[םם]?)\s*$'
    r'|^\S+(?:\s+\S+)?,(?:\s*\S+(?:\s+\S+)?,)*\s*\S+(?:\s+\S+)?\s+ו-?\s*\d+\s*(?:אחרי[םם]?|נוספי[םם]?)\s*$'
    r'|^\S+(?:\s+\S+)?\s+and\s+\d+\s+(?:more|others?)\s*$'
    r'|^\S+(?:\s+\S+)?,(?:\s*\S+(?:\s+\S+)?,)*\s*\S+(?:\s+\S+)?\s+and\s+\d+\s+(?:more|others?)\s*$',
    re.IGNORECASE
)

# רגקס לבדיקת נוכחות אותיות — pre-compiled במקום re.search בכל שורה
_HAS_LETTER_RE = re.compile(r'[a-zA-Zא-ת]')

# רגקס לכיווץ רווחים — pre-compiled במקום re.sub בכל קריאה
_WHITESPACE_RE = re.compile(r'\s+')

# רגקס משולב להסרת תווים בלתי-נראים ו-PUA בפעולה אחת
_INVISIBLE_AND_PUA_RE = re.compile(
    r'[\u200e\u200f\u200b\u200c\u200d\u2060\u2066\u2067\u2068\u2069\ufeff\xad'
    r'\uE000-\uF8FF\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]'
)


@functools.lru_cache(maxsize=512)
def _stable_text_for_hash(text: str) -> str:
    """מנרמל טקסט לחישוב hash יציב — מסיר timestamps, מספרים בודדים, ורעשי UI.

    נרמול עמוק: lowercase, הסרת URLs, תווים בלתי-נראים, מילות רעש UI,
    engagement ו-timestamps. המטרה: אותו פוסט ← אותו hash בין סריקות שונות.
    תוצאות נשמרות ב-cache (lru_cache) — אותו טקסט לא מעובד פעמיים.
    """
    # 1. lowercase — מונע הבדלי case בין רינדורים
    text = text.lower()

    # 2. הסרת URLs — פרמטרי מעקב (fbclid, tracking) משתנים בין טעינות
    text = _URL_RE.sub('', text)

    # 3. הסרת תווים בלתי-נראים (כיווניות, zero-width) ו-PUA — regex משולב אחד
    text = _INVISIBLE_AND_PUA_RE.sub('', text)

    lines = text.split('\n')
    stable = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # בדיקת regex משולבת — digits-only, engagement, dynamic lines בבדיקה אחת
        if _SKIP_LINE_RE.match(s):
            continue
        # דילוג על שורות אמוג'י ללא אותיות — "👍 5", "👍❤️ 12", "😂🔥".
        # דורשים אמוג'י כדי לא לתפוס מספרי טלפון (054-1234567, +972-52-...).
        if _EMOJI_RE.search(s) and not _HAS_LETTER_RE.search(s):
            continue
        # דילוג על מילות רעש UI
        if s in _NOISE_WORDS:
            continue
        # דילוג על בר פעולות משולב ("אהבתי · תגובה · שיתוף")
        if _is_action_bar(s):
            continue
        stable.append(s)
    joined = '\n'.join(stable)
    # הסרת timestamps דינאמיים
    joined = _TIMESTAMP_RE.sub('', joined)
    # כיווץ רווחים מיותרים
    joined = _WHITESPACE_RE.sub(' ', joined).strip()
    return joined


def _extract_name_from_title(page_title: str) -> str | None:
    """מחלץ שם קבוצה מכותרת דף פייסבוק. מחזיר None אם הכותרת לא תקינה."""
    if not page_title:
        return None
    # ניקוי תווי כיווניות (LRM/RLM) שפייסבוק מוסיף
    cleaned = page_title.replace("\u200e", "").replace("\u200f", "").strip()
    # פורמט נפוץ: "שם הקבוצה | Facebook" — לוקחים רק את החלק הראשון
    if " | " in cleaned:
        cleaned = cleaned.split(" | ")[0].strip()
    # פענוח HTML entities (למשל &amp;)
    cleaned = html_mod.unescape(cleaned)
    if (cleaned
            and len(cleaned) > 2
            and cleaned.lower() not in _BAD_TITLES
            and "אינו נתמך" not in cleaned):
        return cleaned
    return None


def load_groups() -> list[dict]:
    """טוען קבוצות מ-DB או ממשתנה סביבה FB_GROUPS.
    סדר עדיפות: DB → FB_GROUPS → רשימה ריקה.
    אין ברירות מחדל — חובה להגדיר קבוצות דרך הפאנל, טלגרם, או FB_GROUPS.
    """
    # ניסיון ראשון — טעינה מ-DB
    try:
        from database import get_db_groups
        db_groups = get_db_groups()
        if db_groups is not None:
            log.info(f"נטענו {len(db_groups)} קבוצות מ-DB")
            return db_groups
    except Exception as e:
        log.debug(f"לא ניתן לטעון קבוצות מ-DB: {e}")

    # fallback — משתנה סביבה
    env_groups = os.environ.get("FB_GROUPS")
    if env_groups:
        groups = []
        for raw_url in env_groups.split(","):
            raw_url = raw_url.strip()
            if not raw_url:
                continue
            # המרה לאתר מובייל
            url = raw_url.replace("www.facebook.com", "m.facebook.com")
            # חילוץ שם הקבוצה מה-URL
            match = re.search(r"/groups/([^/?]+)", url)
            name = match.group(1) if match else url
            groups.append({"name": name, "url": url})
        if groups:
            log.info(f"נטענו {len(groups)} קבוצות ממשתנה סביבה FB_GROUPS")
            return groups
        log.warning("FB_GROUPS מוגדר אבל ריק")

    # אין קבוצות מוגדרות — מחזיר רשימה ריקה
    log.warning("לא הוגדרו קבוצות — יש להוסיף דרך הפאנל, טלגרם, או FB_GROUPS")
    return []

def reload_groups():
    """טוען מחדש את רשימת הקבוצות (נקרא אחרי שינוי דרך טלגרם)."""
    global GROUPS
    GROUPS = load_groups()

GROUPS = load_groups()

async def random_delay(min_sec=1, max_sec=3):
    delay = random.uniform(min_sec, max_sec)
    log.debug(f"השהיה {delay:.1f} שניות")
    await asyncio.sleep(delay)

async def block_heavy_resources(page):
    """חוסם תמונות, וידאו ופונטים כדי לחסוך זיכרון"""
    BLOCKED_TYPES = {"image", "media", "font"}

    async def handle_route(route):
        if route.request.resource_type in BLOCKED_TYPES:
            await route.abort()
        else:
            await route.continue_()

    await page.route("**/*", handle_route)
    log.debug("חסימת משאבים כבדים הופעלה (תמונות, וידאו, פונטים)")

async def save_session(context):
    SESSION_FILE.parent.mkdir(exist_ok=True)
    cookies = await context.cookies()
    SESSION_FILE.write_text(json.dumps(cookies))
    log.info("סשן נשמר")

async def load_session(context):
    if SESSION_FILE.exists():
        cookies = json.loads(SESSION_FILE.read_text())
        await context.add_cookies(cookies)
        log.info(f"סשן נטען ({len(cookies)} cookies)")
        return True
    log.info("לא נמצא סשן שמור")
    return False

async def dismiss_cookie_dialog(page):
    """סוגר דיאלוג cookies אם מופיע"""
    cookie_selectors = [
        "button[data-cookiebanner='accept_button']",
        "button[data-cookiebanner='accept_only_essential_button']",
        "[aria-label='Allow all cookies']",
        "[aria-label='Allow essential and optional cookies']",
        "[aria-label='אישור כל קובצי ה-Cookie']",
        "[title='Allow all cookies']",
        "button[value='Accept All']",
    ]
    for selector in cookie_selectors:
        try:
            btn = await page.query_selector(selector)
            if btn:
                log.info(f"נמצא דיאלוג cookies — לוחץ: {selector}")
                await btn.click()
                await random_delay(1, 2)
                return True
        except Exception:
            continue
    return False

async def _login_desktop(desktop_page, email: str, password: str):
    """לוגין בדסקטופ (www.facebook.com) — form HTML אמיתי עם email+password באותו דף.
    מחזיר את ה-cookies לאחר לוגין מוצלח."""
    log.info("מתחבר לפייסבוק (דסקטופ)...")
    await desktop_page.goto("https://www.facebook.com/login/", wait_until="domcontentloaded")
    await random_delay(1, 3)

    log.debug(f"URL אחרי ניווט (דסקטופ): {desktop_page.url}")

    await dismiss_cookie_dialog(desktop_page)

    email_selectors = ["#email", "input[name='email']", "input[type='email']"]
    pass_selectors = ["#pass", "input[name='pass']", "input[type='password']"]

    # שדה אימייל
    email_field = None
    for selector in email_selectors:
        try:
            email_field = await desktop_page.wait_for_selector(selector, timeout=10000)
            if email_field:
                log.debug(f"[דסקטופ] שדה אימייל נמצא עם: {selector}")
                break
        except Exception:
            continue

    if not email_field:
        try:
            await desktop_page.screenshot(path="data/debug_desktop_login_page.png")
        except Exception:
            pass
        raise Exception("לא נמצא שדה אימייל בדף הלוגין (דסקטופ)")

    await email_field.fill(email)
    await random_delay(0.5, 1)

    # שדה סיסמה — בדסקטופ תמיד באותו דף
    pass_field = None
    for selector in pass_selectors:
        try:
            pass_field = await desktop_page.wait_for_selector(selector, timeout=5000)
            if pass_field:
                log.debug(f"[דסקטופ] שדה סיסמה נמצא עם: {selector}")
                break
        except Exception:
            continue

    if not pass_field:
        try:
            await desktop_page.screenshot(path="data/debug_desktop_no_password.png")
        except Exception:
            pass
        raise Exception("לא נמצא שדה סיסמה בדף הלוגין (דסקטופ)")

    await pass_field.fill(password)
    await random_delay(0.5, 1)

    # שליחת טופס
    log.debug("[דסקטופ] שולח טופס לוגין...")
    login_btn = None
    for sel in ["button[name='login']", "button[type='submit']", "input[type='submit']"]:
        try:
            login_btn = await desktop_page.query_selector(sel)
            if login_btn:
                break
        except Exception:
            continue

    try:
        if login_btn:
            async with desktop_page.expect_navigation(timeout=15000):
                await login_btn.click()
        else:
            async with desktop_page.expect_navigation(timeout=15000):
                await pass_field.press("Enter")
    except Exception:
        await asyncio.sleep(5)

    await random_delay(2, 3)
    return desktop_page.url


async def _login_mobile(page, email: str, password: str):
    """לוגין במובייל (m.facebook.com) — fallback אם אין גישה ל-browser object."""
    log.info("מתחבר לפייסבוק (מובייל)...")
    await page.goto("https://m.facebook.com/login", wait_until="domcontentloaded")
    await random_delay(1, 3)

    log.debug(f"URL אחרי ניווט: {page.url}")

    await dismiss_cookie_dialog(page)

    # שדות לוגין — זהים באתר מובייל
    email_selectors = ["#email", "input[name='email']", "input[type='email']"]
    pass_selectors = [
        "#pass", "input[name='pass']", "input[type='password']",
        "input[autocomplete='current-password']",
        "input[data-sigil~='password-input']",
    ]

    email_field = None
    for selector in email_selectors:
        try:
            email_field = await page.wait_for_selector(selector, timeout=10000)
            if email_field:
                log.debug(f"שדה אימייל נמצא עם: {selector}")
                break
        except Exception:
            continue

    if not email_field:
        try:
            await page.screenshot(path="data/debug_login_page.png")
        except Exception:
            pass
        page_title = await page.title()
        log.error(f"לא נמצא שדה אימייל בדף. כותרת: {page_title}, URL: {page.url}")
        raise Exception("לא נמצא שדה אימייל בדף הלוגין")

    await email_field.fill(email)
    await random_delay(0.5, 1)

    # --- ניסיון ראשון: שדה סיסמה באותו דף ---
    pass_field = None
    for selector in pass_selectors:
        try:
            pass_field = await page.wait_for_selector(selector, timeout=5000)
            if pass_field:
                log.debug(f"שדה סיסמה נמצא עם: {selector}")
                break
        except Exception:
            continue

    # --- לוגין דו-שלבי: אם אין שדה סיסמה, שולחים טופס ומחכים ---
    if not pass_field:
        log.info("שדה סיסמה לא נמצא — שולח טופס כדי לקבל שדה סיסמה")

        # ניסיון 1: Enter על שדה האימייל — הכי אמין, לא תלוי בכפתור ספציפי
        log.debug("שולח Enter על שדה האימייל")
        url_before_enter = page.url
        try:
            async with page.expect_navigation(timeout=15000):
                await email_field.press("Enter")
            log.debug("ניווט לשלב סיסמה התבצע בהצלחה (Enter)")
        except Exception:
            log.debug("לא זוהה ניווט אחרי Enter — ממתין לשינוי DOM...")
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                await asyncio.sleep(3)
        enter_navigated = page.url != url_before_enter

        await random_delay(1, 3)
        await dismiss_cookie_dialog(page)

        # חיפוש שדה סיסמה בדף החדש
        for selector in pass_selectors:
            try:
                pass_field = await page.wait_for_selector(selector, timeout=8000)
                if pass_field:
                    log.debug(f"שדה סיסמה נמצא בשלב 2 עם: {selector}")
                    break
            except Exception:
                continue

        # ניסיון 2: אם Enter לא עזר, מנסים כפתור submit (fallback).
        # חשוב: רק אם ה-Enter לא שינה את הדף! אם כבר ניווטנו לדף סיסמה,
        # לחיצה על submit תשלח טופס עם סיסמה ריקה → סיכון לנעילת חשבון.
        if not pass_field and not enter_navigated:
            log.debug("שדה סיסמה לא נמצא אחרי Enter — מנסה כפתור submit")
            next_btn = None
            for selector in _LOGIN_BTN_SELECTORS:
                try:
                    next_btn = await page.query_selector(selector)
                    if next_btn:
                        log.debug(f"כפתור Next/Login נמצא עם: {selector}")
                        break
                except Exception:
                    continue

            if next_btn:
                try:
                    async with page.expect_navigation(timeout=15000):
                        await next_btn.click()
                    log.debug("ניווט לשלב סיסמה התבצע בהצלחה (כפתור)")
                except Exception:
                    log.debug("לא זוהה ניווט אחרי כפתור — ממתין לשינוי DOM...")
                    try:
                        await page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        await asyncio.sleep(3)

                await random_delay(1, 3)
                await dismiss_cookie_dialog(page)

                for selector in pass_selectors:
                    try:
                        pass_field = await page.wait_for_selector(selector, timeout=8000)
                        if pass_field:
                            log.debug(f"שדה סיסמה נמצא בשלב 2 עם: {selector}")
                            break
                    except Exception:
                        continue

    if not pass_field:
        # דיבאג: לוג תוכן הדף + צילום מסך מקומי
        try:
            page_text = await page.evaluate(
                "() => (document.body?.innerText || '').substring(0, 800)")
            log.error(f"תוכן דף — שדה סיסמה לא נמצא:\n{page_text}")
        except Exception:
            pass
        try:
            await page.screenshot(path="data/debug_no_password_field.png")
        except Exception:
            pass
        page_title = await page.title()
        log.error(f"לא נמצא שדה סיסמה. כותרת: {page_title}, URL: {page.url}")
        raise Exception("לא נמצא שדה סיסמה בדף הלוגין")

    await pass_field.fill(password)
    await random_delay(0.5, 1)

    # שולח טופס — מנסה סלקטורים מרובים לכפתור login, אחרת Enter
    log.debug("שולח טופס לוגין...")
    login_btn = None
    for sel in _LOGIN_BTN_SELECTORS:
        try:
            login_btn = await page.query_selector(sel)
            if login_btn:
                log.debug(f"כפתור login נמצא עם: {sel}")
                break
        except Exception:
            continue

    try:
        if login_btn:
            async with page.expect_navigation(timeout=15000):
                await login_btn.click()
            log.debug("ניווט התבצע בהצלחה (כפתור)")
        else:
            log.debug("לא נמצא כפתור login — שולח Enter")
            async with page.expect_navigation(timeout=15000):
                await pass_field.press("Enter")
            log.debug("ניווט התבצע בהצלחה (Enter)")
    except Exception:
        log.debug("לא זוהה ניווט — ממתין לטעינה ידנית...")
        await asyncio.sleep(5)

    await random_delay(2, 3)
    return page.url


async def login(page, email: str, password: str, browser=None):
    """מתחבר לפייסבוק.

    אם browser מסופק — יוצר context דסקטופ זמני ללוגין ב-www.facebook.com
    (form HTML אמיתי, אמין יותר), מעביר cookies ל-context המובייל, וסוגר.
    אם browser לא מסופק — לוגין ישירות במובייל (fallback, תואם לאחור).

    Cookies של .facebook.com עובדים cross-subdomain (www ↔ m).
    """
    if browser:
        # --- לוגין בדסקטופ והעברת cookies למובייל ---
        log.info("משתמש ב-context דסקטופ ללוגין (www.facebook.com)...")
        desktop_ctx = await browser.new_context(
            user_agent=_DESKTOP_UA,
            viewport={"width": 1280, "height": 720},
            locale="he-IL",
        )
        desktop_page = await desktop_ctx.new_page()
        try:
            post_login_url = await _login_desktop(desktop_page, email, password)

            # בדיקת checkpoint/2FA אחרי לוגין דסקטופ
            if any(m in post_login_url for m in _CHECKPOINT_MARKERS):
                try:
                    await desktop_page.screenshot(path="data/debug_checkpoint.png")
                except Exception:
                    pass
                log.error(f"פייסבוק דורש אימות נוסף (checkpoint): {post_login_url}")
                raise Exception(
                    "פייסבוק דורש אימות נוסף (security checkpoint) — "
                    "היכנס ידנית לחשבון מדפדפן רגיל ואשר את המכשיר. "
                    "ראה debug_checkpoint.png"
                )

            # בדיקה אם עדיין בדף לוגין (לא כולל checkpoint שכבר טופל למעלה)
            if "/login" in post_login_url and not any(m in post_login_url for m in _CHECKPOINT_MARKERS):
                try:
                    await desktop_page.screenshot(path="data/debug_login_failed.png")
                    page_text = await desktop_page.evaluate(
                        "() => (document.body?.innerText || '').substring(0, 800)")
                    log.error(f"תוכן דף לוגין דסקטופ שנכשל:\n{page_text}")
                except Exception:
                    pass
                raise Exception("התחברות נכשלה (דסקטופ) — בדוק אימייל/סיסמה")

            # העברת cookies מ-context דסקטופ ל-context מובייל
            cookies = await desktop_ctx.cookies()
            await page.context.add_cookies(cookies)
            log.info(f"הועברו {len(cookies)} cookies מדסקטופ למובייל")

        finally:
            await desktop_ctx.close()
            log.debug("context דסקטופ נסגר")

    else:
        # --- fallback: לוגין ישירות במובייל ---
        post_login_url = await _login_mobile(page, email, password)

    # --- בדיקות אחרי לוגין (משותף לשני המסלולים) ---
    # אם השתמשנו בדסקטופ, ננווט למובייל כדי לוודא שה-cookies עובדים
    if browser:
        await _goto_with_retry(page, "https://m.facebook.com", wait_until="commit")
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=15000)
        except PlaywrightTimeout:
            log.debug("domcontentloaded לא הגיע תוך 15 שניות — ממשיכים")
        await random_delay(1, 3)
        current_url = page.url
    else:
        current_url = post_login_url

    log.debug(f"URL אחרי לוגין: {current_url}")

    if "/login" in current_url and not any(m in current_url for m in _CHECKPOINT_MARKERS):
        error_msg = None
        for selector in ["div[role='alert']", "#error_box", "div._9ay7", "div.login_error_box"]:
            try:
                el = await page.query_selector(selector)
                if el:
                    error_msg = await el.inner_text()
                    break
            except Exception:
                continue

        # לוג תוכן הדף — לזיהוי CAPTCHA, אבטחה, או שגיאות לא צפויות
        try:
            page_text = await page.evaluate("() => (document.body?.innerText || '').substring(0, 800)")
            log.error(f"תוכן דף לוגין שנכשל:\n{page_text}")
        except Exception:
            pass

        # שמירת צילום מסך מקומי לדיבאג
        try:
            await page.screenshot(path="data/debug_login_failed.png")
        except Exception:
            pass

        if error_msg:
            log.error(f"פייסבוק מציג שגיאה: {error_msg.strip()}")
            raise Exception(f"התחברות נכשלה — פייסבוק אומר: {error_msg.strip()}")
        else:
            log.error(f"התחברות נכשלה — URL עדיין בדף לוגין: {current_url}")
            raise Exception("התחברות נכשלה — בדוק אימייל/סיסמה")

    if any(m in current_url for m in _CHECKPOINT_MARKERS):
        try:
            await page.screenshot(path="data/debug_checkpoint.png")
        except Exception:
            pass
        log.error(f"פייסבוק דורש אימות נוסף (checkpoint): {current_url}")
        raise Exception(
            "פייסבוק דורש אימות נוסף (security checkpoint) — "
            "היכנס ידנית לחשבון מדפדפן רגיל ואשר את המכשיר. "
            "ראה debug_checkpoint.png"
        )

    # אימות נוסף: פייסבוק מובייל מפנה מ-/login ל-/ — בדיקת URL לא מספיקה.
    # אם שדות הלוגין עדיין מוצגים, ההתחברות לא הצליחה באמת
    # (ייתכן CAPTCHA, סיסמה שגויה, חסימה, או שינוי ב-UI).
    still_has_form = None
    try:
        still_has_form = await page.query_selector("input[name='email']")
    except Exception:
        pass

    if still_has_form:
        try:
            page_text = await page.evaluate(
                "() => (document.body?.innerText || '').substring(0, 800)")
            log.error(f"תוכן דף — טופס לוגין עדיין מוצג:\n{page_text}")
        except Exception:
            pass
        try:
            await page.screenshot(path="data/debug_login_failed.png")
        except Exception:
            pass
        raise Exception(
            "התחברות נכשלה — טופס לוגין עדיין מוצג "
            "(ייתכן CAPTCHA או סיסמה שגויה)")

    log.info(f"מחובר בהצלחה — URL: {current_url}")

def extract_post_id(url: str) -> str:
    patterns = [
        r"/posts/(\d+)",
        r"story_fbid=(\d+)",
        r"/permalink/(\d+)",
        r"pfbid(\w+)",
        r"/share/p/(\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url[-20:]

def _is_post_url(href: str) -> bool:
    """בודק אם URL הוא קישור לפוסט ספציפי (לא לקבוצה בלבד)."""
    post_patterns = ["/posts/", "/permalink/", "story_fbid", "/story.php", "multi_permalinks", "/p/", "pfbid", "/share/p/"]
    return any(p in href for p in post_patterns)

def _extract_id_from_data_attrs(attrs_json: str) -> str | None:
    """מחלץ מזהה פוסט מתוך JSON של data attributes (data-ft, data-store)."""
    try:
        data = json.loads(attrs_json)
    except (json.JSONDecodeError, TypeError):
        return None
    # סדר עדיפות — מזהים מוכרים ב-data-ft של פייסבוק מובייל
    for key in ("mf_story_key", "top_level_post_id", "tl_objid"):
        val = data.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return None


# מונה פנימי לסטטיסטיקת חילוץ URL
_extraction_stats: dict[str, int] = {"success": 0, "fallback": 0}
# דגל: scrape_group זיהה מצב wui → scrape_all צריך להתחבר מחדש
_session_needs_refresh = False
# דגל לעצירת סריקה באמצע — נבדק בלולאת הקבוצות
_stop_scan_requested = False


def request_stop_scan():
    """נקרא מטלגרם (/stop) — מבקש עצירת סריקה באמצע."""
    global _stop_scan_requested
    _stop_scan_requested = True

# JavaScript לסריקת כל הדף — מוצא את כל קישורי הפוסטים לפני עיבוד אלמנטים.
# מאפשר התאמה גם כשהאלמנט עצמו קטן מדי ואין בו לינקים.
_PAGE_SCAN_JS = """() => {
    const PP = ['/posts/', '/permalink/', 'story_fbid', '/story.php',
                'multi_permalinks', 'pfbid', '/p/', '/share/p/'];
    const isPU = h => PP.some(p => h.includes(p));
    const pfx = h => h.startsWith('http') ? h : 'https://m.facebook.com' + h;

    const allLinks = [...document.querySelectorAll('a[href]')];
    const postLinks = [];
    const seenUrls = new Set();

    for (const a of allLinks) {
        const href = a.getAttribute('href') || '';
        if (!isPU(href)) continue;
        const url = pfx(href);
        if (seenUrls.has(url)) continue;
        seenUrls.add(url);

        // טיפוס בעץ למציאת מכולת הפוסט (טקסט > 100 תווים)
        let node = a;
        let text = '';
        for (let i = 0; i < 30; i++) {
            if (!node.parentElement || node.parentElement === document.body) break;
            node = node.parentElement;
            text = (node.innerText || '').trim();
            if (text.length > 100) break;
        }

        postLinks.push({url, text: text.substring(0, 1000)});
    }

    return {
        totalLinks: allLinks.length,
        postLinkCount: postLinks.length,
        postLinks,
        sampleHrefs: allLinks.slice(0, 15).map(a =>
            (a.getAttribute('href') || '').substring(0, 200))
    };
}"""

# JavaScript לטיפוס בעץ ה-DOM — קריאה אחת שבודקת עד 15 רמות.
# מחפש: קישורי פוסט, לינקי timestamp (עברית + אנגלית), ajaxify, data-ft/data-store.
_CLIMB_DOM_JS = """(el, fallbackUrl) => {
    const PP = ['/posts/', '/permalink/', 'story_fbid',
                '/story.php', 'multi_permalinks', 'pfbid', '/p/', '/share/p/'];
    const tsHe = /\\d+\\s*(שע|דק|יו|ימ|שני|חוד)/;
    const tsEn = /^\\d+\\s*[hdmwHDMW]$/;
    const tsExact = new Set(['עכשיו', 'אתמול', 'just now', 'yesterday']);
    const isPU = h => PP.some(p => h.includes(p));
    const isTs = t => {
        t = (t || '').trim();
        return tsHe.test(t) || tsEn.test(t) || tsExact.has(t.toLowerCase());
    };
    const pfx = h => h.startsWith('http') ? h : 'https://m.facebook.com' + h;
    const gm = fallbackUrl.match(/\\/groups\\/([^\\/?]+)/);
    const gid = gm ? gm[1] : null;

    let node = el;
    for (let i = 0; i < 30; i++) {
        node = node.parentElement;
        if (!node) break;

        const links = node.querySelectorAll('a[href]');
        for (const a of links) {
            const h = a.getAttribute('href') || '';
            if (isPU(h)) return JSON.stringify({url: pfx(h), lvl: i+1, how: 'post_link'});
        }
        for (const a of links) {
            const h = a.getAttribute('href') || '';
            const t = a.textContent || '';
            if (isTs(t) && h.includes('/groups/') && h !== fallbackUrl) {
                const f = pfx(h);
                if (f.length > fallbackUrl.length + 5)
                    return JSON.stringify({url: f, lvl: i+1, how: 'timestamp'});
            }
        }

        const ajx = node.querySelectorAll('[ajaxify]');
        for (const ae of ajx) {
            const v = ae.getAttribute('ajaxify') || '';
            if (isPU(v)) return JSON.stringify({url: pfx(v), lvl: i+1, how: 'ajaxify'});
        }

        if (gid) {
            for (const attr of ['data-ft', 'data-store']) {
                const found = node.querySelector('[' + attr + ']');
                const target = found || (node.hasAttribute && node.hasAttribute(attr) ? node : null);
                if (target) {
                    try {
                        const d = JSON.parse(target.getAttribute(attr));
                        for (const k of ['mf_story_key', 'top_level_post_id', 'tl_objid']) {
                            if (d[k] && String(d[k]).trim())
                                return JSON.stringify({
                                    url: 'https://m.facebook.com/groups/' + gid
                                         + '/permalink/' + String(d[k]).trim() + '/',
                                    lvl: i+1, how: attr
                                });
                        }
                    } catch(e) {}
                }
            }
        }
    }
    return null;
}"""


def get_extraction_stats() -> dict[str, int]:
    """מחזיר עותק של סטטיסטיקת חילוץ URL (לשימוש בטסטים ולוגים)."""
    return dict(_extraction_stats)


def reset_extraction_stats():
    """מאפס סטטיסטיקת חילוץ URL."""
    _extraction_stats["success"] = 0
    _extraction_stats["fallback"] = 0


async def _extract_post_url(el, fallback_url: str, *,
                            page_urls: list[dict] | None = None) -> str:
    """מחלץ URL לפוסט מתוך אלמנט — מנסה מספר אסטרטגיות.

    page_urls — קישורי פוסטים שנאספו בסריקת דף (לאסטרטגיה 7).
    """
    # אסטרטגיה 1: סלקטורים ישירים לקישורי פוסטים
    direct_selectors = [
        "a[href*='/posts/']",
        "a[href*='story_fbid']",
        "a[href*='permalink']",
        "a[href*='/story.php']",
        "a[href*='multi_permalinks']",
        "a[href*='pfbid']",
        "a[href*='/p/']",
        "a[href*='/share/p/']",
    ]
    for sel in direct_selectors:
        link = await el.query_selector(sel)
        if link:
            href = await link.get_attribute("href")
            if href:
                log.debug(f"חילוץ URL — אסטרטגיה 1 (סלקטור ישיר): {sel}")
                _extraction_stats["success"] += 1
                return href if href.startswith("http") else f"https://m.facebook.com{href}"

    # אסטרטגיה 2: סריקת כל הלינקים — מחפש כל URL שמכיל מזהה פוסט
    all_links = await el.query_selector_all("a[href]")
    for a in all_links:
        href = await a.get_attribute("href") or ""
        if _is_post_url(href):
            log.debug(f"חילוץ URL — אסטרטגיה 2 (סריקת לינקים): {href[:80]}")
            _extraction_stats["success"] += 1
            return href if href.startswith("http") else f"https://m.facebook.com{href}"

    # אסטרטגיה 3: לינק חותמת זמן — במובייל, הזמן ("12 שעות") הוא לינק לפוסט.
    # קישור timestamp בתוך קבוצה הוא כמעט תמיד קישור לפוסט הספציפי,
    # גם אם ה-URL לא מכיל מזהה פוסט מוכר (כמו pfbid, /posts/ וכו')
    for a in all_links:
        href = await a.get_attribute("href") or ""
        text = (await a.inner_text()).strip() if a else ""
        # חותמת זמן בעברית: "12 שעות", "3 ימים", "דקה", "אתמול" וכו'
        # חותמת זמן באנגלית: "12h", "3d", "1m", "Just now", "Yesterday"
        is_timestamp = (
            re.search(r'\d+\s*(שע|דק|יו|ימ|שני|חוד)', text)
            or text in ("עכשיו", "אתמול")
            or re.search(r'^\d+\s*[hdmwHDMW]$', text)
            or text.lower() in ("just now", "yesterday")
        )
        if is_timestamp and "/groups/" in href and href != fallback_url:
            full = href if href.startswith("http") else f"https://m.facebook.com{href}"
            # ודא שהקישור ארוך יותר מ-URL הקבוצה (כלומר מכיל מזהה נוסף)
            if len(full) > len(fallback_url) + 5:
                log.debug(f"חילוץ URL — אסטרטגיה 3 (timestamp): {full[:80]}")
                _extraction_stats["success"] += 1
                return full

    # אסטרטגיה 4: חילוץ מזהה פוסט מ-data attributes (data-ft, data-store)
    # ב-m.facebook.com, אלמנטים מכילים JSON עם מזהי פוסט
    group_id_match = re.search(r"/groups/([^/?]+)", fallback_url)
    if group_id_match:
        group_id = group_id_match.group(1)
        for attr in ("data-ft", "data-store"):
            try:
                attr_val = await el.get_attribute(attr)
                if attr_val:
                    post_id = _extract_id_from_data_attrs(attr_val)
                    if post_id:
                        url = f"https://m.facebook.com/groups/{group_id}/permalink/{post_id}/"
                        log.debug(f"חילוץ URL — אסטרטגיה 4 ({attr}): {url}")
                        _extraction_stats["success"] += 1
                        return url
            except Exception:
                continue

        # חיפוש data-ft/data-store גם באלמנטים פנימיים
        for attr in ("data-ft", "data-store"):
            try:
                inner = await el.query_selector(f"[{attr}]")
                if inner:
                    attr_val = await inner.get_attribute(attr)
                    if attr_val:
                        post_id = _extract_id_from_data_attrs(attr_val)
                        if post_id:
                            url = f"https://m.facebook.com/groups/{group_id}/permalink/{post_id}/"
                            log.debug(f"חילוץ URL — אסטרטגיה 4 ({attr} פנימי): {url}")
                            _extraction_stats["success"] += 1
                            return url
            except Exception:
                continue

    # אסטרטגיה 5: חיפוש מזהה פוסט ב-id/aria של האלמנט עצמו
    try:
        el_id = await el.get_attribute("id") or ""
        # פייסבוק לפעמים שם ID כמו "mall_post_123456" או "u_0_xx_yy"
        id_match = re.search(r'(\d{10,})', el_id)
        if id_match and group_id_match:
            post_id = id_match.group(1)
            url = f"https://m.facebook.com/groups/{group_id_match.group(1)}/permalink/{post_id}/"
            log.debug(f"חילוץ URL — אסטרטגיה 5 (element id): {url}")
            _extraction_stats["success"] += 1
            return url
    except Exception:
        pass

    # אסטרטגיה 6: חיפוש מקיף בעץ ה-DOM — קריאת JavaScript אחת.
    # עולים עד 30 רמות ובודקים:
    # קישורי פוסט (<a href>), לינקי timestamp, ajaxify, data-ft/data-store.
    # קריאה אחת במקום מספר round-trips → מהיר ויציב יותר.
    try:
        found = await el.evaluate(_CLIMB_DOM_JS, fallback_url)
        if found:
            data = json.loads(found)
            log.debug(f"חילוץ URL — אסטרטגיה 6 (DOM רמה {data['lvl']}, {data['how']}): {data['url'][:80]}")
            _extraction_stats["success"] += 1
            return data["url"]
    except Exception:
        pass

    # אסטרטגיה 7: התאמה לקישור מסריקת דף (page-level scan).
    # כשהאלמנט קטן מדי (div[dir='auto']) ואין בו לינקים, מחפשים התאמה
    # לפי תוכן טקסט — אם הטקסט של האלמנט מופיע בתוך מכולת פוסט שיש לה URL.
    if page_urls:
        try:
            el_text = (await el.inner_text()).strip()
            # שימוש ב-80 התווים הראשונים כמפתח חיפוש (מספיק לזיהוי ייחודי)
            search_key = el_text[:80]
            if len(search_key) >= 30:
                for entry in page_urls:
                    if search_key in entry.get('text', ''):
                        url = entry['url']
                        log.debug(f"חילוץ URL — אסטרטגיה 7 (page-level): {url[:80]}")
                        _extraction_stats["success"] += 1
                        return url
        except Exception:
            pass

    # כל האסטרטגיות נכשלו
    _extraction_stats["fallback"] += 1

    log.debug(f"חילוץ URL נכשל — משתמש ב-fallback URL: {fallback_url[:60]}")
    return fallback_url

async def scrape_group(page, group: dict, seen_checker=None) -> list[dict]:
    posts = []
    reset_extraction_stats()
    group_name = group["name"]
    log.info(f">>> סורק קבוצה: {group_name}")
    log.debug(f"URL: {group['url']}")

    try:
        log.debug("ניווט לדף הקבוצה...")
        await _goto_with_retry(page, group["url"], wait_until="commit")
        # commit חוזר מהר אבל ה-DOM עדיין לא מוכן — ממתינים ל-domcontentloaded
        # עם timeout קצר. אם פג — ממשיכים בכל זאת (עדיף מ-135 שניות תקיעה).
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=15000)
        except PlaywrightTimeout:
            log.debug("domcontentloaded לא הגיע תוך 15 שניות — ממשיכים")
        current_url = page.url
        log.debug(f"הגענו ל: {current_url}")

        if "login" in current_url:
            log.warning("הועברנו לדף לוגין — ייתכן שהסשן פג!")
            return posts

        # זיהוי overlay לוגין — פייסבוק מציג "פתיחת האפליקציה / התחברות"
        # בלי redirect, ה-URL נשאר של הקבוצה אבל התוכן חסום.
        # סלקטורים ספציפיים ל-overlay בלבד — לא כוללים a[href*='/login'] כי זה
        # מופיע גם בניווט של דפים רגילים ויגרום לזיהוי שווא.
        overlay_dismissed = False
        try:
            login_overlay = await page.query_selector(
                "[data-sigil='login_interstitial'], "
                "[data-sigil='m_login_upsell'], "
                "#login_popup_cta_form, "
                "div[id*='login_upsell']"
            )
            if login_overlay:
                # ננסה לסגור את ה-overlay
                close_btn = await page.query_selector(
                    "[data-sigil='login_interstitial_dismiss'], "
                    "button[aria-label='Close'], "
                    "button[aria-label='סגור']"
                )
                if close_btn:
                    await close_btn.click()
                    log.info("סגרנו overlay לוגין — ממשיכים סריקה")
                    await random_delay(1, 2)
                    overlay_dismissed = True
                else:
                    log.warning("זוהה overlay לוגין שלא ניתן לסגירה — ייתכן שהסשן פג!")
                    return posts
        except Exception:
            pass

        # Fallback: זיהוי overlay לוגין לפי טקסט ("פתיחת האפליקציה / התחברות")
        # כשהסלקטורים למעלה לא תופסים — ייתכן שפייסבוק שינה HTML.
        # לא רץ אם כבר סגרנו overlay בהצלחה — הטקסט עלול להישאר ב-DOM לרגע.
        if not overlay_dismissed and await _has_login_overlay(page):
            log.warning("זוהה overlay לוגין (לפי טקסט) — ייתכן שהסשן פג!")
            return posts

        # חילוץ שם הקבוצה מכותרת הדף (אמין יותר מסלקטורים בתוך ה-DOM)
        group_name = group["name"]
        try:
            page_title = await page.title()
            extracted = _extract_name_from_title(page_title)
            if extracted:
                group_name = extracted
                # עדכון שם הקבוצה ב-DB כדי שישתמר לסבבים הבאים
                if group_name != group["name"]:
                    try:
                        from database import update_group_name
                        update_group_name(group["url"], group_name)
                        log.info(f"שם קבוצה עודכן: {group['name']} → {group_name}")
                    except Exception as e:
                        log.warning(f"נכשל עדכון שם קבוצה ב-DB: {e}")
            else:
                # warning רק אם השם עדיין לא עודכן (ספרות בלבד) — אחרת זה מצב רגיל
                if group["name"].isdigit():
                    log.warning(f"לא הצלחתי לחלץ שם קבוצה מכותרת הדף: '{page_title}'")
        except Exception as e:
            log.warning(f"שגיאה בחילוץ כותרת דף לקבוצה {group['url']}: {e}")

        await random_delay(2, 4)

        # זיהוי מצב wui — פייסבוק מציג רק לינקי "פתח באפליקציה" (סשן פג).
        # חשוב: הבדיקה רצה *אחרי* השהיה כדי לתת ל-React זמן לרנדר.
        # בדיקה כפולה: (1) רוב הלינקים הם wui, (2) אין אלמנטי פוסטים בדף.
        if await _is_wui_page(page):
            log.debug("זוהה מצב wui — ממתין 5 שניות נוספות לרינדור...")
            await asyncio.sleep(5)
            if await _is_wui_page(page):
                # בדיקה נוספת: אם יש אלמנטי פוסטים, הדף לא באמת wui-only
                has_posts = await page.query_selector(
                    "article, div[data-ft], div[role='article']")
                if has_posts:
                    log.debug(
                        "יש אלמנטי פוסטים למרות לינקי wui — ממשיכים (לא wui-only)")
                else:
                    log.warning(
                        "דף קבוצה במצב wui — רק לינקי 'פתח באפליקציה', "
                        "אין קישורי פוסט. ייתכן שהסשן פג!"
                    )
                    global _session_needs_refresh
                    _session_needs_refresh = True
                    return posts
            else:
                log.debug("מצב wui נעלם אחרי המתנה — ממשיכים")

        # גלילה חכמה — גוללים עד שמגיעים לפוסט שכבר נראה בסריקה קודמת
        MAX_SCROLLS = 10
        MAX_POSTS = 30
        SELECTORS = "article, div[data-ft], div[role='article']"
        FALLBACK_SELECTORS = "div[dir='auto']"

        reached_known = False
        consecutive_known = 0
        KNOWN_THRESHOLD = 3  # כמה פוסטים מוכרים ברצף לפני עצירה (מונע עצירה על פוסט מוצמד)
        seen_texts = set()
        skipped_short = 0
        page_post_urls = []  # קישורי פוסטים שנאספו בסריקת דף (page-level scan)

        log.debug("מתחיל גלילה חכמה...")

        for scroll_i in range(MAX_SCROLLS + 1):
            # בסיבוב הראשון לא גוללים — מעבדים מה שכבר נטען
            if scroll_i > 0:
                await page.evaluate("window.scrollBy(0, Math.random() * 300 + 200)")
                log.debug(f"גלילה {scroll_i}/{MAX_SCROLLS}")
                await random_delay(1, 2)

            post_elements = await page.query_selector_all(SELECTORS)
            if not post_elements:
                post_elements = await page.query_selector_all(FALLBACK_SELECTORS)
                if post_elements and scroll_i == 0:
                    log.warning(f"סלקטור ראשי לא מצא פוסטים — נפלנו ל-fallback ({len(post_elements)} אלמנטים)")

            if not post_elements and scroll_i == 0:
                log.warning(f"לא נמצאו אלמנטים בכלל בקבוצה: {group_name}")
                try:
                    await page.screenshot(path=f"data/debug_{group_name[:20]}.png")
                except Exception:
                    pass
                break

            # סריקת דף — מציאת כל קישורי הפוסטים לשימוש באסטרטגיה 7.
            # רץ בכל סיבוב גלילה כי תוכן חדש נטען אחרי גלילה.
            try:
                scan = await page.evaluate(_PAGE_SCAN_JS)
                existing = {e['url'] for e in page_post_urls}
                new_count = 0
                for entry in scan.get('postLinks', []):
                    if entry['url'] not in existing:
                        page_post_urls.append(entry)
                        existing.add(entry['url'])
                        new_count += 1
                if scroll_i == 0:
                    log.debug(
                        f"סריקת דף — {scan['totalLinks']} לינקים, "
                        f"{scan['postLinkCount']} קישורי פוסטים"
                    )
                    if scan['postLinkCount'] == 0 and scan.get('sampleHrefs'):
                        log.debug(f"דוגמאות hrefs (ללא קישורי פוסט): {scan['sampleHrefs'][:5]}")
                elif new_count > 0:
                    log.debug(f"סריקת דף — {new_count} קישורים חדשים (סה\"כ {len(page_post_urls)})")
            except Exception as e:
                log.debug(f"שגיאה בסריקת דף: {e}")

            # מעבדים את כל האלמנטים ומדלגים על כאלה שכבר ראינו לפי תוכן.
            # לא משתמשים במעקב לפי אינדקס כי ה-DOM יכול להשתנות בגלילה
            # (virtualization, החלפת selector) מה שגורם לדילוג על אלמנטים חדשים.
            prev_seen_count = len(seen_texts)

            for el in post_elements:
                try:
                    text = await asyncio.wait_for(el.inner_text(), timeout=5)
                    # הסרת תגיות HTML שדלפו (כולל <img> עם base64)
                    text = re.sub(r'<[^>]+>', '', text)
                    text = text.strip()

                    if text in seen_texts:
                        continue
                    seen_texts.add(text)

                    if len(text) < 50:
                        skipped_short += 1
                        continue

                    # חילוץ שם המפרסם + קישור לפרופיל
                    author = ""
                    author_url = ""
                    try:
                        author_info = await asyncio.wait_for(
                            el.evaluate("""(el) => {
                                let name = '', url = '';
                                const s = el.querySelector('strong');
                                const h = el.querySelector('h3');
                                const nameEl = s || h;
                                if (nameEl) {
                                    name = (nameEl.innerText || '').trim();
                                    // אסטרטגיה 1: הלינק לפרופיל עוטף את ה-strong/h3
                                    const a = nameEl.closest('a') || nameEl.querySelector('a');
                                    if (a && a.href) url = a.href;
                                }
                                // אסטרטגיה 2: לינק פרופיל על תמונה או ליד השם (header)
                                // ב-m.facebook.com השם לא תמיד עטוף ב-<a> — הלינק על האווטר
                                if (!url) {
                                    const skip = /\/(groups|posts|stor(y|ies)(\.php)?|permalink|wui|shar(e|ing)|photos?(\.php)?|videos?(\.php)?|watch|hashtag|events?|pages?|marketplace|gaming|reels?|multi_permalinks)(\/|$|\?)/;
                                    const allLinks = el.querySelectorAll('a[href]');
                                    for (const link of allLinks) {
                                        const href = link.getAttribute('href') || '';
                                        if (!href || href === '#') continue;
                                        // דילוג על לינקים פנימיים של פייסבוק שאינם פרופיל
                                        // pfbid נבדק בנפרד כי ה-ID מודבק ישירות: /pfbid02Kabc123
                                        if (skip.test(href) || href.includes('/pfbid')) continue;
                                        // דילוג על לינקים חיצוניים (lm.facebook.com/l.php)
                                        if (href.includes('/l.php') || href.includes('lm.facebook.com')) continue;
                                        // לינק פרופיל: /profile.php?id=, /username, או פשוט /שם
                                        if (href.includes('/profile.php') || href.includes('facebook.com/')) {
                                            url = link.href;
                                            break;
                                        }
                                        // לינק יחסי שמתחיל ב-/ ולא ב-// (לא פרוטוקול)
                                        if (href.startsWith('/') && !href.startsWith('//') && href.length > 1) {
                                            url = link.href;
                                            break;
                                        }
                                    }
                                }
                                return {name, url};
                            }"""),
                            timeout=3,
                        )
                        author = author_info.get("name", "")
                        author_url = author_info.get("url", "")
                        if author and not author_url:
                            log.debug(f"[AUTHOR] שם נמצא ({author}) אבל אין קישור לפרופיל")
                    except Exception:
                        pass

                    post_url = group["url"]
                    try:
                        post_url = await _extract_post_url(
                            el, group["url"], page_urls=page_post_urls
                        )
                    except Exception:
                        pass

                    has_real_url = post_url != group["url"]

                    if has_real_url:
                        post_id = extract_post_id(post_url)
                    else:
                        # fallback URL = URL הקבוצה → ID זהה לכל הפוסטים בקבוצה.
                        # משתמשים ב-hash של תוכן מנורמל כ-ID ייחודי לכל פוסט.
                        # חשוב: הטקסט מנורמל (ללא timestamps, לייקים וכו')
                        # כדי שה-hash יהיה יציב בין סריקות.
                        stable = _stable_text_for_hash(text)
                        if not stable:
                            # הטקסט כולו timestamps/UI → לא תוכן אמיתי, מדלגים
                            skipped_short += 1
                            continue
                        post_id = "c_" + hashlib.md5(stable.encode()).hexdigest()[:16]

                    # בדיקה אם הגענו לפוסט מסריקה קודמת.
                    # ה-hash מבוסס תוכן מנורמל (ללא timestamps) ולכן יציב בין סריקות.
                    # דורשים רצף של KNOWN_THRESHOLD פוסטים מוכרים לפני עצירה,
                    # כדי למנוע עצירה מוקדמת בגלל פוסט מוצמד (pinned) בראש הפיד.
                    if seen_checker and seen_checker(post_id):
                        consecutive_known += 1
                        if consecutive_known >= KNOWN_THRESHOLD:
                            reached_known = True
                            log.info(f"הגענו ל-{KNOWN_THRESHOLD} פוסטים מוכרים ברצף — עוצרים גלילה (אחרי {scroll_i} גלילות)")
                            break
                        log.debug(f"פוסט מוכר ({consecutive_known}/{KNOWN_THRESHOLD}), ממשיכים")
                        continue
                    else:
                        consecutive_known = 0

                    posts.append({
                        "id": post_id,
                        "content": text,
                        "url": post_url,
                        "group": group_name,
                        "group_url": group["url"],
                        "has_real_url": has_real_url,
                        "author": author,
                        "author_url": author_url,
                    })

                    if len(posts) >= MAX_POSTS:
                        log.debug(f"הגענו למקסימום {MAX_POSTS} פוסטים — עוצרים")
                        reached_known = True
                        break

                except Exception as e:
                    log.debug(f"שגיאה בעיבוד אלמנט: {e}")
                    continue

            if reached_known:
                break

            if scroll_i > 0 and len(seen_texts) == prev_seen_count:
                log.debug("אין אלמנטים חדשים אחרי גלילה — עוצרים")
                break

        if not reached_known:
            log.debug(f"לא הגענו לפוסט מוכר — נעצרנו אחרי {scroll_i} גלילות")

        log.debug(f"סטטיסטיקה — דולגו: {skipped_short} קצרים")

    except PlaywrightTimeout as e:
        log.error(f"טיימאאוט בקבוצה: {group['name']} — {e}")
    except Exception as e:
        log.error(f"שגיאה בקבוצה {group['name']}: {e}", exc_info=True)

    # סטטיסטיקת חילוץ URL לקבוצה
    stats = get_extraction_stats()
    total_attempts = stats["success"] + stats["fallback"]
    if total_attempts > 0:
        rate = stats["success"] / total_attempts * 100
        real_url_posts = sum(1 for p in posts if p.get("has_real_url"))
        log.info(
            f"חילוץ URL — הצלחה: {stats['success']}/{total_attempts} ({rate:.0f}%) | "
            f"פוסטים עם URL אמיתי: {real_url_posts}/{len(posts)}"
        )
        if stats["fallback"] > 0:
            log.warning(
                f"חילוץ URL נכשל ב-{stats['fallback']} פוסטים — "
                f"fallback ל-URL קבוצה (התנהגות צפויה ב-m.facebook.com)"
            )

    log.info(f"<<< נמצאו {len(posts)} פוסטים ב-{group_name}")
    return posts

async def scrape_all(email: str, password: str, seen_checker=None,
                     on_group_done=None) -> list[dict]:
    """סורק את כל הקבוצות ומחזיר רשימת פוסטים.

    on_group_done — callback אופציונלי שנקרא אחרי כל קבוצה:
        on_group_done(group_name, group_index, total_groups, posts_found)
    """
    all_posts = []

    # איפוס דגל עצירה מיד בתחילת הסריקה — לפני כל פעולה async,
    # כדי למנוע race condition עם /stop שנשלח בזמן הפעלת דפדפן/לוגין.
    global _stop_scan_requested
    _stop_scan_requested = False

    if not GROUPS:
        log.warning("אין קבוצות מוגדרות — מדלג על סריקה. הוסף קבוצות דרך הפאנל, טלגרם, או FB_GROUPS")
        return all_posts

    log.info(f"מתחיל סריקה של {len(GROUPS)} קבוצות (מובייל)")

    async with async_playwright() as p:
        log.debug("מפעיל דפדפן Chromium...")
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-extensions",
                "--disable-background-networking",
                "--disable-default-apps",
                "--disable-sync",
                "--disable-translate",
                "--no-first-run",
                "--disable-component-update",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-software-rasterizer",
                "--renderer-process-limit=1",
                "--js-flags=--max-old-space-size=128",
            ]
        )
        log.debug("דפדפן הופעל בהצלחה")

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            viewport={"width": 360, "height": 640},
            locale="he-IL",
            service_workers="block",  # מונע מ-SW של פייסבוק לתקוע את טעינת הדף
        )
        page = await context.new_page()
        await block_heavy_resources(page)

        session_loaded = await load_session(context)

        if session_loaded:
            log.debug("בודק תקינות סשן...")
            await _goto_with_retry(page, "https://m.facebook.com", wait_until="commit")
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=15000)
            except PlaywrightTimeout:
                log.debug("domcontentloaded לא הגיע תוך 15 שניות — ממשיכים")
            await random_delay(1, 3)
            await dismiss_cookie_dialog(page)
            is_wui = await _is_wui_page(page)
            if "login" in page.url or await _has_login_overlay(page) or is_wui:
                reason = "wui-only (רק לינקי 'פתח באפליקציה')" if is_wui else "login redirect/overlay"
                log.warning(f"סשן פג ({reason}) — מנקה cookies ומתחבר מחדש")
                await context.clear_cookies()
                await login(page, email, password, browser=browser)
                await save_session(context)
            else:
                log.info("סשן תקין — ממשיך")
        else:
            await login(page, email, password, browser=browser)
            await save_session(context)

        global _session_needs_refresh
        _session_needs_refresh = False

        for i, group in enumerate(GROUPS, 1):
            if _stop_scan_requested:
                log.info("סריקה נעצרה לפי בקשת המשתמש (/stop)")
                break

            log.info(f"--- קבוצה {i}/{len(GROUPS)} ---")
            posts = await scrape_group(page, group, seen_checker=seen_checker)

            # אם scrape_group זיהה מצב wui — הסשן פג. מתחברים מחדש ומנסים שוב.
            if _session_needs_refresh:
                _session_needs_refresh = False
                log.warning("סשן פג זוהה בקבוצה — מנקה cookies ומתחבר מחדש...")
                try:
                    await context.clear_cookies()
                    await login(page, email, password, browser=browser)
                    await save_session(context)
                    posts = await scrape_group(page, group, seen_checker=seen_checker)
                    if _session_needs_refresh:
                        # גם אחרי re-login עדיין wui — בעיה אחרת, עוצרים
                        _session_needs_refresh = False
                        log.error("סשן פג גם אחרי התחברות מחדש — עוצרים סריקה")
                        break
                except Exception as e:
                    log.error(f"התחברות מחדש נכשלה: {e}")
                    break

            all_posts.extend(posts)

            # עדכון callback התקדמות (אם סופק)
            if on_group_done:
                try:
                    on_group_done(group["name"], i, len(GROUPS), len(posts))
                except Exception:
                    pass

            # ניקוי זיכרון — מנווט לדף ריק כדי לשחרר את ה-DOM
            try:
                await page.goto("about:blank")
                log.debug("ניווט ל-about:blank לשחרור זיכרון")
            except Exception:
                pass

            if i < len(GROUPS):
                log.debug("השהיה בין קבוצות...")
                await random_delay(3, 8)

        await browser.close()
        log.debug("דפדפן נסגר")

    log.info(f"סריקה הסתיימה — סה\"כ {len(all_posts)} פוסטים מ-{len(GROUPS)} קבוצות")
    return all_posts


