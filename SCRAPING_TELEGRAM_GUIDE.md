# מדריך: בניית כלי סריקת אתר + התראות טלגרם

מדריך מקיף לבניית פרויקט שסורק אתר בקביעות, מזהה נתונים רלוונטיים, ושולח התראות לטלגרם.
מבוסס על לקחים ופתרונות מפרויקט אמיתי שרץ בפרודקשן על VM עם 512MB RAM.

---

## תוכן עניינים

1. [ארכיטקטורה כללית](#1-ארכיטקטורה-כללית)
2. [מבנה קבצים מומלץ](#2-מבנה-קבצים-מומלץ)
3. [שכבת סריקה (Scraper)](#3-שכבת-סריקה-scraper)
4. [שכבת סינון וסיווג](#4-שכבת-סינון-וסיווג)
5. [שכבת התראות (Notifier)](#5-שכבת-התראות-notifier)
6. [שכבת נתונים (Database)](#6-שכבת-נתונים-database)
7. [אורקסטרציה (Main Loop)](#7-אורקסטרציה-main-loop)
8. [בקרה דרך טלגרם](#8-בקרה-דרך-טלגרם)
9. [Deduplication — מניעת כפילויות](#9-deduplication--מניעת-כפילויות)
10. [ניהול זיכרון וביצועים](#10-ניהול-זיכרון-וביצועים)
11. [Deployment](#11-deployment)
12. [טסטים](#12-טסטים)
13. [טעויות נפוצות ופתרונות](#13-טעויות-נפוצות-ופתרונות)
14. [Checklist לפרויקט חדש](#14-checklist-לפרויקט-חדש)

---

## 1. ארכיטקטורה כללית

### עקרון הפיפליין

כל פרויקט סריקה+התראות עובד באותו פיפליין:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Scrape  │ →  │  Dedup   │ →  │  Filter  │ →  │ Classify │ →  │  Notify  │
│ (אתר)   │    │ (DB)     │    │ (מילים)  │    │ (AI)     │    │(טלגרם)   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

כל שלב מפחית את כמות הפריטים — כך חוסכים זמן, API calls, וספאם למשתמש.

### עקרון ההפרדה

כל שכבה בקובץ נפרד. יתרונות:
- קל להחליף שכבה (למשל Playwright → Selenium, או טלגרם → Slack)
- קל לטסט כל שכבה בנפרד
- באגים מבודדים לקובץ אחד

---

## 2. מבנה קבצים מומלץ

```
project/
├── main.py           # אורקסטרציה + לולאת בקרה
├── scraper.py        # סריקת האתר
├── classifier.py     # סיווג (AI או כללים)
├── notifier.py       # שליחת הודעות
├── database.py       # שכבת נתונים (SQLite)
├── logger.py         # לוגר פשוט
├── tests.py          # טסטים
├── requirements.txt  # תלויות
├── Dockerfile        # הרצה בקונטיינר
├── render.yaml       # deploy (Render/Railway)
├── data/             # תיקייה לנתונים מקומיים
│   ├── leads.db      # SQLite DB
│   └── session.json  # session cookies (אם רלוונטי)
└── .env              # משתני סביבה (לא ב-git!)
```

---

## 3. שכבת סריקה (Scraper)

### בחירת טכנולוגיה

| טכנולוגיה | מתי להשתמש | זיכרון |
|-----------|-----------|--------|
| **requests + BeautifulSoup** | אתר סטטי, API גלוי | ~20MB |
| **Playwright** | אתר דינמי, צריך JS, צריך login | ~150-300MB |
| **Selenium** | דומה ל-Playwright, ישן יותר | ~200-400MB |
| **API ישיר** | יש API רשמי/לא רשמי | ~10MB |

> **כלל אצבע:** התחל עם requests. עבור ל-Playwright רק אם חייבים JS rendering או login.

### דפוס בסיסי — requests

```python
import requests
from bs4 import BeautifulSoup

async def scrape_page(url: str) -> list[dict]:
    """סורק דף ומחזיר רשימת פריטים."""
    response = await asyncio.to_thread(
        requests.get, url, timeout=15, headers=HEADERS
    )
    soup = BeautifulSoup(response.text, 'html.parser')

    items = []
    for element in soup.select('.item-selector'):
        items.append({
            "id": extract_id(element),
            "text": element.get_text(strip=True),
            "url": element.find('a')['href'],
            "has_real_url": True,
        })
    return items
```

### דפוס מתקדם — Playwright

```python
from playwright.async_api import async_playwright

async def scrape_with_browser(url: str) -> list[dict]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 360, "height": 640},  # מובייל = קל יותר
            user_agent="Mozilla/5.0 ..."
        )
        page = await context.new_page()

        # חסימת משאבים כבדים
        await page.route("**/*", lambda route: (
            route.abort() if route.request.resource_type
            in {"image", "media", "font"}
            else route.continue_()
        ))

        await page.goto(url, wait_until="domcontentloaded")

        # גלילה לטעינת תוכן דינמי
        items = await smart_scroll_and_extract(page)

        await browser.close()
        return items
```

### גלילה חכמה — עקרונות

**לא** לגלול מספר קבוע של פעמים. במקום זה:

```python
async def smart_scroll(page, seen_checker) -> list[dict]:
    """גולל עד שנגמר תוכן חדש או הגענו למגבלה."""
    seen_texts = set()          # מעקב לפי תוכן, לא אינדקס!
    consecutive_known = 0       # רצף של פריטים מוכרים
    all_items = []

    for scroll_num in range(MAX_SCROLLS):
        elements = await page.query_selector_all('.item')
        new_in_scroll = 0

        for el in elements:
            text = await el.inner_text()
            if text in seen_texts:
                continue
            seen_texts.add(text)

            item = extract_item(el, text)

            # בדיקת dedup רק אם יש ID אמיתי
            if item["has_real_url"] and seen_checker(item["id"]):
                consecutive_known += 1
                if consecutive_known >= KNOWN_THRESHOLD:
                    return all_items  # הגענו לפוסטים ישנים
                continue

            consecutive_known = 0  # אפס את הרצף
            new_in_scroll += 1
            all_items.append(item)

        if new_in_scroll == 0:
            break  # גלילה לא הביאה תוכן חדש

        await page.evaluate("window.scrollBy(0, 1500)")
        await asyncio.sleep(random.uniform(1, 3))

    return all_items
```

#### למה מעקב לפי תוכן ולא אינדקס?

אתרים רבים (פייסבוק, טוויטר, LinkedIn) משתמשים ב-**DOM וירטואלי** — אלמנטים נוספים ומוסרים תוך כדי גלילה. אם עוקבים לפי אינדקס (`elements[5:]`), הרפרנסים נשברים. תמיד לעקוב לפי **set של תוכן שכבר נראה**.

#### למה צריך רצף של פריטים מוכרים?

פריט מוצמד (pinned) בראש הפיד כבר ב-DB → אם בודקים פריט בודד, הגלילה נעצרת מיד עם 0 תוצאות. תמיד לדרוש **רצף** (3+) של פריטים מוכרים.

### שימור Session

```python
import json
from pathlib import Path

SESSION_FILE = Path("data/session.json")

async def save_session(context):
    """שומר cookies לשימוש חוזר — חוסך login חוזר."""
    cookies = await context.cookies()
    SESSION_FILE.parent.mkdir(exist_ok=True)
    SESSION_FILE.write_text(json.dumps(cookies))

async def load_session(context):
    """טוען cookies מסשן קודם."""
    if SESSION_FILE.exists():
        cookies = json.loads(SESSION_FILE.read_text())
        await context.add_cookies(cookies)
```

### זיהוי Session פג — כללים

**סף מחמיר + בדיקה כפולה** — כי false positive שובר את כל הסריקה:

```python
async def is_session_expired(page) -> bool:
    """בודק אם הסשן פג. שמרני — false positive שובר הכל."""
    # בדיקה 1: URL הופנה לדף login
    if "/login" in page.url:
        return True

    # בדיקה 2: סימנים מרובים (לא אחד בלבד!)
    indicators = await page.evaluate("""() => {
        const links = [...document.querySelectorAll('a[href]')];
        const loginLinks = links.filter(a =>
            a.href.includes('/login') || a.href.includes('/signin'));
        const hasContent = document.querySelectorAll('.item, article').length > 0;
        return {
            loginRatio: loginLinks.length / Math.max(links.length, 1),
            hasContent: hasContent,
            totalLinks: links.length
        };
    }""")

    # רק אם יחס גבוה של לינקי login וגם אין תוכן
    if indicators["loginRatio"] > 0.75 and not indicators["hasContent"]:
        return True

    return False
```

**הכלל הזהב:** מנגנוני זיהוי סשן חייבים להיות **שמרניים**. false positive שובר את כל הסריקה (re-login נכשל → 0 תוצאות), false negative רק מפספס סבב אחד.

### חילוץ ID — אסטרטגיות מרובות

כשאי אפשר לחלץ URL/ID ישיר, צריך אסטרטגיות fallback:

```python
def extract_item_id(element, page_url: str) -> tuple[str, bool]:
    """מחלץ ID מפריט. מחזיר (id, has_real_id).

    has_real_id = True רק כשחולץ ID ייחודי אמיתי.
    Fallback = hash של תוכן — לא סומכים עליו ל-dedup.
    """
    # אסטרטגיה 1: href ישיר
    link = element.find('a', href=True)
    if link and '/item/' in link['href']:
        return extract_id_from_url(link['href']), True

    # אסטרטגיה 2: data attribute
    if element.get('data-id'):
        return element['data-id'], True

    # אסטרטגיה 3: hash של תוכן (fallback)
    text = element.get_text(strip=True)
    content_hash = hashlib.md5(text[:200].encode()).hexdigest()[:12]
    return f"hash_{content_hash}", False
```

**חשוב:** כש-`has_real_id = False`, ה-dedup checker צריך להתעלם מהפריט — אחרת כל הפריטים שנופלים ל-fallback יקבלו ID דומה ויחסמו אחד את השני.

### השהיות אנושיות

```python
async def random_delay(min_sec=1, max_sec=3):
    """השהיה אקראית למניעת חסימה."""
    delay = random.uniform(min_sec, max_sec)
    await asyncio.sleep(delay)
```

---

## 4. שכבת סינון וסיווג

### פיפליין סינון — מהזול ליקר

```
raw posts
  │
  ├─ [1] Force Send (מילות "שלח תמיד")     ← חינם, עוקף הכל
  ├─ [2] Dedup (seen_posts DB)               ← חינם
  ├─ [3] Age Filter (גיל הפוסט)             ← חינם
  ├─ [4] Block Filter (מילים חסומות)         ← חינם
  ├─ [5] Pre-filter (מילות מפתח)             ← חינם
  ├─ [6] Content Dedup (hash תוכן)           ← חינם
  └─ [7] AI Classification                   ← $$$  ← רק מה שעבר הכל
```

כל שלב חינמי מפחית את כמות הקריאות ל-AI. סדר קריטי!

### סינון מילות מפתח

```python
def passes_keyword_filter(text: str, keywords: list[str]) -> bool:
    """בודק אם הטקסט מכיל לפחות מילת מפתח אחת.

    !!חשוב!! רשימה ריקה = אין סינון = מעביר הכל.
    any() על רשימה ריקה מחזיר False — חייבים guard.
    """
    if not keywords:
        return True  # אין סינון = הכל עובר
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)

def is_blocked(text: str, block_words: list[str]) -> bool:
    """בודק אם הטקסט מכיל מילה חסומה.

    רשימה ריקה = אין חסימה = הכל עובר.
    """
    if not block_words:
        return False  # אין חסימה = שום דבר לא חסום
    text_lower = text.lower()
    return any(w in text_lower for w in block_words)
```

> **כלל: רשימה ריקה = אין סינון = מעביר הכל.** `any()` על `[]` מחזיר `False` — בלי guard, רשימת מילות מפתח ריקה תחסום *הכל* בשקט.

### סיווג עם AI — באצ'ים + Fallback

```python
import openai

# רשימת מודלים לפי סדר עדיפות — auto-fallback כשמודל deprecated
MODEL_PRIORITY = ["gpt-4.1-mini", "gpt-4o-mini"]
active_model = MODEL_PRIORITY[0]

def classify_batch(items: list[dict], batch_size=5) -> list[dict]:
    """מסווג פריטים בבאצ'ים — חוסך API calls.

    כלל: תוצאות באצ' נאספות ברשימה זמנית.
    רק אחרי הצלחה מלאה מוסיפים ל-all_results.
    """
    all_results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        try:
            response = call_api(batch)
            batch_results = parse_response(response)

            # וולידציה: אורך תשובה = אורך באצ'
            if len(batch_results) != len(batch):
                raise ValueError("אורך תשובה לא תואם")

            # בדיקת null — אלמנט null שובר alignment
            if any(r is None for r in batch_results):
                raise ValueError("אלמנט null בתשובה")

            # הצלחה מלאה — מוסיפים הכל
            all_results.extend(batch_results)

        except Exception:
            # fallback: סיווג בודד פריט-פריט
            for item in batch:
                try:
                    result = classify_single(item)
                    all_results.append(result)
                except Exception:
                    all_results.append({"relevant": False, "reason": "error"})

    return all_results
```

#### למה רשימה זמנית לבאצ'?

אם אלמנט בתשובת JSON לא תקין, הלולאה שמוסיפה תוצאות זורקת exception אחרי הוספה **חלקית**. ה-fallback מוסיף את כל הבאצ' שוב → `all_results` ארוך מ-`items` → `zip()` מתאים תוצאות לפריטים לא נכונים.

### Auto-Fallback למודל

```python
def call_api_with_fallback(messages: list) -> dict:
    """קריאת API עם auto-fallback למודל הבא אם deprecated."""
    model = get_active_model()

    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
    except openai.APIError as e:
        if is_model_deprecated(e):  # 404 או 410
            next_model = rotate_model(model)
            if next_model:
                return client.chat.completions.create(
                    model=next_model, messages=messages, temperature=0.1
                )
        raise

def is_model_deprecated(e: openai.APIError) -> bool:
    if hasattr(e, "status_code") and e.status_code in (404, 410):
        return True
    msg = str(e).lower()
    return any(h in msg for h in [
        "deprecated", "does not exist", "model not found"
    ])
```

### פירוש JSON מ-API

```python
import json
import re

def parse_json_response(raw: str) -> dict | list:
    """מפרש JSON מתשובת API — כולל unwrap של markdown blocks.

    חייבים להשתמש בפונקציה הזו בכל מקום — לא לשכפל לוגיקה.
    """
    text = raw.strip()

    # Unwrap markdown code block
    md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    return json.loads(text)
```

---

## 5. שכבת התראות (Notifier)

### שליחת הודעות טלגרם

```python
import requests

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_message(
    text: str,
    *,
    chat_id: str | None = None,
    parse_mode: str | None = None,
    disable_web_page_preview: bool = False,
) -> bool:
    """שולח הודעה לטלגרם. מחזיר True בהצלחה."""
    if not BOT_TOKEN or not (chat_id or CHAT_ID):
        log.error("חסרים TELEGRAM_BOT_TOKEN או TELEGRAM_CHAT_ID")
        return False

    target = str(chat_id or CHAT_ID)
    payload = {
        "chat_id": target,
        "text": text,
        "disable_web_page_preview": disable_web_page_preview,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json=payload,
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        log.error(f"שגיאה בשליחה לטלגרם: {e}")
        return False
```

### שליחה מתוך async

```python
# חשוב! requests.post חוסם את ה-event loop
# חייבים לעטוף ב-asyncio.to_thread

async def send_lead(item: dict, reason: str):
    text = format_lead(item, reason)
    success = await asyncio.to_thread(send_message, text)
    if success:
        save_lead(item["id"], item["text"])
```

### הפרדת ערוצי הודעות

```python
# ערוץ רגיל = ללקוח (לידים בלבד)
# ערוץ שגיאות = למפתח (שגיאות טכניות)
ERROR_CHAT_ID = os.environ.get("ERROR_CHAT_ID")

async def send_error(message: str):
    """שולח שגיאה לערוץ המפתח — לא ללקוח."""
    target = ERROR_CHAT_ID or CHAT_ID
    await asyncio.to_thread(send_message, message, chat_id=target)
```

### ניקוי תוכן לפני שליחה

```python
import re

def clean_content(text: str) -> str:
    """מנקה תוכן מ-Unicode פרטי ורעשי UI."""
    # הסרת HTML שדלף
    text = re.sub(r'<[^>]+>', '', text)
    # הסרת Private Use Area characters
    text = re.sub(r'[\uE000-\uF8FF]', '', text)
    text = re.sub(r'[\U000F0000-\U0010FFFF]', '', text)
    # ניקוי שורות ריקות מיותרות
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
```

---

## 6. שכבת נתונים (Database)

### למה SQLite?

- **אפס תלויות** — חלק מ-Python standard library
- **קובץ בודד** — קל לגיבוי ו-deploy
- **מספיק לפרויקטים קטנים-בינוניים** — עד מאות אלפי רשומות
- **thread-safe** עם connection pool

### סכמת DB מומלצת

```python
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "leads.db"
_local = threading.local()

def _get_conn() -> sqlite3.Connection:
    """Thread-local connection pool — כל thread מקבל חיבור משלו."""
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH)
    return _local.conn

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with _get_conn() as conn:
        # פריטים שנראו — dedup ראשי
        conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_items (
                item_id TEXT PRIMARY KEY,
                source TEXT,
                seen_at TEXT
            )
        """)

        # פריטים שנשלחו — dedup + היסטוריה
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sent_items (
                item_id TEXT PRIMARY KEY,
                source TEXT,
                content TEXT,
                reason TEXT,
                sent_at TEXT,
                content_hash TEXT DEFAULT ''
            )
        """)

        # קונפיגורציה — key-value גמיש
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
```

### Dedup Functions

```python
def is_seen(item_id: str) -> bool:
    row = _get_conn().execute(
        "SELECT 1 FROM seen_items WHERE item_id = ?", (item_id,)
    ).fetchone()
    return row is not None

def mark_seen(item_id: str, source: str):
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO seen_items VALUES (?, ?, ?)",
            (item_id, source, _now_str())
        )

def is_content_hash_sent(content_hash: str) -> bool:
    """בודק אם תוכן דומה כבר נשלח — תופס כפילויות שה-ID שלהן השתנה."""
    row = _get_conn().execute(
        "SELECT 1 FROM sent_items WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return row is not None
```

### קונפיגורציה לפי סוג — לא לפי טבלה

```python
# לעולם לא לבדוק "האם הטבלה הוגדרה" ברמת שם הטבלה!
# אם יש סוגים שונים באותה טבלה (למשל keywords עם type=pre_filter ו-type=block),
# בדיקה ברמת הטבלה תגרום לבאגים.

# נכון: מפתח לכל סוג בטבלת _config
def is_type_configured(type_name: str) -> bool:
    """בודק אם סוג מסוים הוגדר ע"י המשתמש."""
    row = _get_conn().execute(
        "SELECT 1 FROM _config WHERE key = ?",
        (f"configured_{type_name}",)
    ).fetchone()
    return row is not None

# דוגמה: keywords_pre_filter vs keywords_block
# כל אחד מסומן בנפרד ב-_config
```

### ניקוי רשומות ישנות

```python
def cleanup_old(days: int = 30):
    """מוחק רשומות ישנות — מונע גדילת DB אינסופית."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with _get_conn() as conn:
        conn.execute("DELETE FROM seen_items WHERE seen_at < ?", (cutoff,))
```

---

## 7. אורקסטרציה (Main Loop)

### לולאה ראשית — Async

```python
import asyncio
import gc
import signal
import sys

async def main():
    init_db()

    # health check server
    start_health_server()

    # לולאת בקרה טלגרם — רצה במקביל
    control_task = asyncio.create_task(telegram_control_loop())
    # חשוב! שמירה במשתנה — אחרת GC יאסוף את ה-task

    scan_event = asyncio.Event()

    while True:
        if is_quiet_hours():
            await asyncio.sleep(60)
            continue

        try:
            await run_scan_cycle()
        except Exception as e:
            log.error(f"שגיאה בסריקה: {e}")
            await send_error(f"שגיאה: {e}")

        # המתנה עד הסריקה הבאה (או scan מיידי מטלגרם)
        try:
            await asyncio.wait_for(
                scan_event.wait(),
                timeout=INTERVAL_MINUTES * 60
            )
            scan_event.clear()
        except asyncio.TimeoutError:
            pass  # timeout = הגיע זמן לסריקה רגילה
```

### מחזור סריקה

```python
async def run_scan_cycle():
    """מחזור סריקה מלא — scrape → filter → classify → notify."""

    # 1. סריקה
    all_items = await scrape_all_sources()
    log.info(f"נמצאו {len(all_items)} פריטים")

    # 2. שחרור זיכרון דפדפן לפני סיווג
    gc.collect()

    # 3. Dedup — סינון פריטים שכבר נראו
    new_items = [i for i in all_items if not is_seen(i["id"])]

    # 4. סימון כנראו (גם אם לא רלוונטיים)
    for item in new_items:
        mark_seen(item["id"], item["source"])

    # 5. סינון — מילות מפתח, מילים חסומות
    filtered = [i for i in new_items
                if passes_keyword_filter(i["text"], PRE_FILTER_KEYWORDS)
                and not is_blocked(i["text"], BLOCK_KEYWORDS)]

    # 6. Content dedup — כפילויות לפי תוכן
    unique = []
    for item in filtered:
        ch = content_hash(item["text"])
        if not is_content_hash_sent(ch):
            item["_content_hash"] = ch
            unique.append(item)

    # 7. סיווג AI (הכי יקר — רק על מה שעבר הכל)
    if unique:
        results = classify_batch(unique)
        for item, result in zip(unique, results):
            if result["relevant"]:
                await send_lead(item, result["reason"])
                save_lead(item, result["reason"])

    log.info(f"סריקה הסתיימה: {len(all_items)} נסרקו, "
             f"{len(new_items)} חדשים, {len(unique)} לסיווג")
```

### שעות שקטות

```python
from datetime import time

def parse_quiet_hours(spec: str) -> tuple[time, time] | None:
    """מפרש '02:00-07:00' לטווח שעות שקטות."""
    if not spec:
        return None
    match = re.match(r'(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})', spec)
    if not match:
        return None
    start = time(int(match.group(1)), int(match.group(2)))
    end = time(int(match.group(3)), int(match.group(4)))
    return (start, end)

def is_quiet_now(quiet_hours: tuple[time, time] | None) -> bool:
    """בודק אם עכשיו שעות שקטות. תומך במעבר חצות."""
    if not quiet_hours:
        return False
    start, end = quiet_hours
    now = datetime.now(tz).time()

    if start <= end:
        # טווח רגיל: 02:00-07:00
        return start <= now < end
    else:
        # מעבר חצות: 22:00-06:00
        return now >= start or now < end
```

### סיגנלים ו-Graceful Shutdown

```python
_shutting_down = False

def handle_signal(sig, frame):
    global _shutting_down
    _shutting_down = True
    log.info("מתחיל shutdown מסודר...")
    sys.exit(0)

# רק ב-main thread
if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
```

---

## 8. בקרה דרך טלגרם

### Polling Loop

```python
async def telegram_control_loop():
    """לולאת polling לפקודות טלגרם."""
    offset = 0

    while True:
        try:
            updates = await asyncio.to_thread(
                requests.get,
                f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                params={"offset": offset, "timeout": 30},
                timeout=35,
            )

            data = updates.json()
            if not data.get("ok"):
                await asyncio.sleep(5)
                continue

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                await handle_update(update)

        except Exception as e:
            # backoff מעריכי על 409 (conflict בין instances)
            if "409" in str(e):
                await asyncio.sleep(random.uniform(5, 15))
            else:
                await asyncio.sleep(5)
```

### טיפול בפקודות

```python
async def handle_update(update: dict):
    message = update.get("message", {})
    text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id")

    if text == "/status":
        status = get_status_text()
        await asyncio.to_thread(send_message, status, chat_id=chat_id)

    elif text == "/scan":
        scan_event.set()  # מפעיל סריקה מיידית
        await asyncio.to_thread(
            send_message, "סריקה מתחילה...", chat_id=chat_id
        )

    elif text.startswith("/add_keyword "):
        word = text[13:].strip()
        add_keyword(word, "pre_filter")
        await asyncio.to_thread(
            send_message, f"מילת מפתח נוספה: {word}", chat_id=chat_id
        )
```

### כפתורים (Inline Keyboard)

```python
def send_with_buttons(text: str, buttons: list[list[dict]], chat_id=None):
    """שולח הודעה עם כפתורים אינטראקטיביים."""
    payload = {
        "chat_id": chat_id or CHAT_ID,
        "text": text,
        "reply_markup": {"inline_keyboard": buttons},
    }
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json=payload,
        timeout=10,
    )

# דוגמה לתפריט ראשי
def main_menu_buttons():
    return [
        [{"text": "סריקה עכשיו", "callback_data": "scan"}],
        [{"text": "סטטוס", "callback_data": "status"}],
        [{"text": "הגדרות", "callback_data": "settings"}],
    ]
```

---

## 9. Deduplication — מניעת כפילויות

### הגנה תלת-שכבתית

```
שכבה 1: Item ID
  └── "ראינו את הפריט הזה?" → seen_items table

שכבה 2: Content Hash (לפני שליחה)
  └── "שלחנו תוכן דומה?" → sent_items.content_hash

שכבה 3: Item ID (לפני שליחה)
  └── "שלחנו את הפריט הזה?" → sent_items.item_id
```

### Content Hash יציב

הבעיה: תוכן דינמי (תגובות, לייקים, timestamps) משנה את ה-hash בין סריקות.

```python
import hashlib
import re

def stable_text_for_hash(text: str) -> str:
    """מנרמל טקסט כך שאותו פריט ← אותו hash בין סריקות."""
    text = text.lower()

    # הסרת URLs (פרמטרי tracking משתנים)
    text = re.sub(r'https?://\S+', '', text)

    # הסרת תווים בלתי נראים
    text = re.sub(r'[\u200e\u200f\u200b\u200c\u200d\u2060\ufeff]', '', text)

    # הסרת Private Use Area (אייקונים שמשתנים בין רנדורים)
    text = re.sub(r'[\uE000-\uF8FF\U000F0000-\U0010FFFD]', '', text)

    lines = text.split('\n')
    stable = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # דילוג על מספרים בודדים (ספירת לייקים)
        if re.match(r'^\d[\d,. ]*$', s):
            continue
        # דילוג על מילות UI
        if s in NOISE_WORDS:
            continue
        # דילוג על engagement ("5 תגובות", "3 שיתופים")
        if ENGAGEMENT_RE.match(s):
            continue
        stable.append(s)

    result = ' '.join(stable)
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def content_dedup_hash(text: str) -> str:
    """Hash של 150 התווים הראשונים בלבד.

    למה 150 ולא הכל?
    - ליבת הפריט (כותרת + תחילת גוף) תמיד בהתחלה
    - תוכן דינמי (תגובות, engagement) בסוף
    - חיתוך ל-150 מבטיח hash יציב

    למה לא 200? בפריטים קצרים (~180 תווים), 200 כולל חלק מתגובות.
    """
    normalized = stable_text_for_hash(text)
    core = normalized[:150]
    return hashlib.md5(core.encode()).hexdigest()
```

---

## 10. ניהול זיכרון וביצועים

### עקרונות ל-VM קטן (512MB)

1. **אתר מובייל** — אם יש גרסת מובייל, תשתמש בה. חוסך ~60% זיכרון.

2. **חסימת משאבים כבדים:**
```python
async def block_heavy_resources(page):
    BLOCKED = {"image", "media", "font"}
    await page.route("**/*", lambda route: (
        route.abort() if route.request.resource_type in BLOCKED
        else route.continue_()
    ))
```

3. **ניווט ל-about:blank בין דפים** — משחרר DOM:
```python
# אחרי סריקת כל דף
await page.goto("about:blank")
```

4. **gc.collect() בנקודות מפתח:**
```python
# אחרי סגירת דפדפן, לפני סיווג AI
await browser.close()
gc.collect()
```

5. **דפדפן אחד, context חדש לכל סריקה:**
```python
# לא לפתוח דפדפנים מרובים
# context חדש = ניקוי זיכרון ללא restart של chromium
```

### Async נכון

```python
# לעולם לא requests.post() ישירות ב-async function
# זה חוסם את ה-event loop

# שגוי:
async def bad():
    requests.post(url, data=data)  # חוסם!

# נכון:
async def good():
    await asyncio.to_thread(requests.post, url, data=data)
```

---

## 11. Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# תלויות מערכת ל-Chromium (רק אם משתמשים ב-Playwright)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libdbus-1-3 libxkbcommon0 libatspi2.0-0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

RUN playwright install chromium

COPY . .
RUN mkdir -p data

ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "main.py"]
```

### requirements.txt מינימלי

```
# סריקה דינמית
playwright==1.52.0

# סיווג AI
openai==1.82.0

# התראות טלגרם
requests==2.32.3

# פאנל ווב (אופציונלי)
flask==3.1.0
```

### משתני סביבה

```bash
# חובה
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
OPENAI_API_KEY=your_openai_key

# סריקה (אם צריך login)
SITE_EMAIL=your@email.com
SITE_PASSWORD=your_password

# אופציונלי
INTERVAL_MINUTES=10             # מרווח בין סריקות
TIMEZONE=Asia/Jerusalem         # אזור זמן
QUIET_HOURS=02:00-07:00         # שעות שקטות
LOG_LEVEL=DEBUG                 # רמת לוג
ERROR_CHAT_ID=dev_chat_id       # ערוץ שגיאות נפרד
```

### Health Check

```python
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *args):
        pass  # שקט

def start_health_server(port=8080):
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
```

---

## 12. טסטים

### מבנה

```python
import pytest
from unittest.mock import patch, MagicMock

# טסטים לכל שכבה בנפרד

class TestKeywordFilter:
    def test_empty_keywords_passes_all(self):
        """רשימה ריקה = אין סינון = הכל עובר."""
        assert passes_keyword_filter("any text", []) is True

    def test_matching_keyword(self):
        assert passes_keyword_filter("need a developer", ["developer"]) is True

    def test_no_match(self):
        assert passes_keyword_filter("hello world", ["developer"]) is False

class TestDedup:
    def test_stable_hash_ignores_engagement(self):
        """אותו פוסט עם engagement שונה = אותו hash."""
        text1 = "looking for dev\n5 comments"
        text2 = "looking for dev\n8 comments"
        assert stable_text_for_hash(text1) == stable_text_for_hash(text2)

    def test_content_hash_150_chars(self):
        """hash מבוסס על 150 תווים ראשונים בלבד."""
        base = "a" * 150
        text1 = base + " some comment"
        text2 = base + " different comment"
        assert content_dedup_hash(text1) == content_dedup_hash(text2)

class TestClassifier:
    @patch("classifier.client")
    def test_batch_fallback_on_error(self, mock_client):
        """באצ' שנכשל → fallback לסיווג בודד."""
        mock_client.chat.completions.create.side_effect = Exception("API error")
        # fallback צריך לעבוד...

class TestQuietHours:
    def test_midnight_wrap(self):
        """22:00-06:00 — מעבר חצות."""
        quiet = (time(22, 0), time(6, 0))
        # 23:00 = שקט
        # 05:00 = שקט
        # 07:00 = לא שקט
```

### הרצה

```bash
python -m pytest tests.py -v
```

---

## 13. טעויות נפוצות ופתרונות

### 1. קריאות sync חוסמות event loop

```python
# שגוי — חוסם async
async def send():
    requests.post(...)

# נכון
async def send():
    await asyncio.to_thread(requests.post, ...)
```

### 2. רשימה ריקה חוסמת הכל

```python
# שגוי — any([]) == False → כל הפריטים נחסמים
def filter(text, keywords):
    return any(kw in text for kw in keywords)

# נכון
def filter(text, keywords):
    if not keywords:
        return True  # אין סינון = הכל עובר
    return any(kw in text for kw in keywords)
```

### 3. מעקב לפי אינדקס בגלילה

```python
# שגוי — DOM וירטואלי משנה אינדקסים
for i in range(last_index, len(elements)):
    process(elements[i])

# נכון — מעקב לפי תוכן
seen_texts = set()
for el in elements:
    text = el.text
    if text in seen_texts:
        continue
    seen_texts.add(text)
    process(el)
```

### 4. Fallback ID יוצר כפילויות

```python
# שגוי — כל הפריטים ללא URL מקבלים ID דומה
id = extract_url(el) or page_url  # כולם page_url!

# נכון — flag שמסמן ID אמיתי
id, has_real_id = extract_item_id(el, page_url)
if has_real_id:
    check_dedup(id)  # רק על ID אמיתי
```

### 5. באצ' חלקי שובר alignment

```python
# שגוי — exception באמצע מוסיף חלקית
for result in batch_response:
    all_results.append(result)  # נופל באמצע!
# fallback מוסיף הכל שוב → כפילויות

# נכון — רשימה זמנית
temp = []
for result in batch_response:
    temp.append(result)
all_results.extend(temp)  # רק אחרי הצלחה מלאה
```

### 6. בדיקת configuration ברמת טבלה

```python
# שגוי — שני סוגים באותה טבלה, בדיקה אחת
def is_configured(table):
    return table in sqlite_sequence  # block הוגדר → גם pre_filter "מוגדר"

# נכון — בדיקה לכל סוג בנפרד
def is_configured(type_name):
    return config.get(f"configured_{type_name}") is not None
```

### 7. נרמול URL ללא סכמה

```python
# שגוי — מחליפים דומיין לפני בדיקת סכמה
url = url.replace("www.", "m.")  # "www.site.com" → "m.site.com"
if not url.startswith("http"):
    url = "https://" + url  # "https://m.site.com" ✓

# אבל: "site.com/path/m.site.com/..." → prefix כפול!

# נכון — סכמה קודם, אח"כ דומיין
if not url.startswith("http"):
    url = "https://" + url
url = url.replace("www.", "m.")
```

### 8. 409 Conflict בטלגרם

```python
# כשיש שתי instances שעושות polling → 409
# פתרון: backoff מעריכי
try:
    get_updates()
except Conflict409:
    await asyncio.sleep(random.uniform(5, 15))
```

### 9. Task נאסף ע"י GC

```python
# שגוי — task ללא רפרנס נאסף
asyncio.create_task(telegram_loop())

# נכון — שמירה במשתנה
_control_task = asyncio.create_task(telegram_loop())
```

### 10. WUI/Session detection אגרסיבי מדי

```python
# שגוי — סימן אחד = "סשן פג"
if any_login_link_found:
    re_login()  # false positive → login נכשל → 0 תוצאות

# נכון — סף גבוה + בדיקה כפולה
if login_ratio > 0.75 AND no_content_elements AND link_count >= 3:
    re_login()
```

---

## 14. Checklist לפרויקט חדש

### שלב 1: תכנון
- [ ] מה סורקים? (אתר, API, RSS)
- [ ] צריך login? (Playwright) או לא (requests)
- [ ] מה מחפשים? (מילות מפתח, AI, כללים)
- [ ] לאן שולחים? (טלגרם, Slack, email)
- [ ] כמה זיכרון יש ב-VM?

### שלב 2: שלד
- [ ] `scraper.py` — סריקה בסיסית + חילוץ פריטים
- [ ] `database.py` — סכמה + dedup
- [ ] `notifier.py` — שליחת הודעות
- [ ] `main.py` — לולאה בסיסית
- [ ] `logger.py` — לוגר
- [ ] `tests.py` — טסטים בסיסיים

### שלב 3: סינון
- [ ] Pre-filter — מילות מפתח (חינמי)
- [ ] Block filter — מילים חסומות
- [ ] Content dedup — hash תוכן
- [ ] AI classification — רק על מה שעבר הכל

### שלב 4: בקרה
- [ ] לולאת טלגרם — פקודות בסיסיות (/status, /scan)
- [ ] שעות שקטות
- [ ] Health check endpoint

### שלב 5: Production
- [ ] Dockerfile
- [ ] משתני סביבה
- [ ] ניהול זיכרון (חסימת משאבים, gc.collect)
- [ ] טיפול בשגיאות (retry, fallback)
- [ ] ניקוי DB ישן
- [ ] לוגים מספיקים לדיבוג

### שלב 6: שיפורים
- [ ] Auto-fallback למודל AI
- [ ] כפתורים אינטראקטיביים בטלגרם
- [ ] מעקב עלויות API
- [ ] פאנל ווב (Flask)
- [ ] Force-send keywords (עוקף AI)
- [ ] ערוץ שגיאות נפרד למפתח

---

## סיכום עקרונות זהב

1. **פיפליין מהזול ליקר** — סינון חינמי לפני AI
2. **Dedup תלת-שכבתי** — ID + content hash + בדיקה לפני שליחה
3. **Hash יציב** — נרמול עמוק, 150 תווים ראשונים בלבד
4. **רשימה ריקה = אין סינון** — guard על `any([])`
5. **מעקב לפי תוכן, לא אינדקס** — DOM וירטואלי שובר אינדקסים
6. **Async בכל מקום** — `asyncio.to_thread()` לקריאות sync
7. **באצ' = רשימה זמנית** — extend רק אחרי הצלחה מלאה
8. **Session detection שמרני** — false positive שובר הכל
9. **Flag על ID אמיתי** — fallback ID לא מתאים ל-dedup
10. **גרסת מובייל + חסימת משאבים** — חוסך 60%+ זיכרון
