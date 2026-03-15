# קוד ריוויו ושיפורים — Facebook Leads Bot

**תאריך:** 2026-03-06
**סוקר:** Claude Code

---

## תוכן עניינים

1. [סקירה כללית](#סקירה-כללית)
2. [בעיות קוד ושיפורים](#בעיות-קוד-ושיפורים)
3. [בעיות אבטחה](#בעיות-אבטחה)
4. [שיפורי ביצועים וזיכרון](#שיפורי-ביצועים-וזיכרון)
5. [שיפורי ארכיטקטורה](#שיפורי-ארכיטקטורה)
6. [הצעות לפיצ'רים חדשים](#הצעות-לפיצ'רים-חדשים)
7. [סיכום עדיפויות](#סיכום-עדיפויות)

---

## סקירה כללית

הפרויקט בנוי טוב ומתועד היטב. ה-CLAUDE.md מפורט ומלמד על היסטוריה ארוכה של תיקוני באגים ולמידה מטעויות. 339 טסטים מכסים את הלוגיקה המרכזית. עם זאת, יש מקום לשיפורים משמעותיים בכמה תחומים.

**מה טוב:**
- תיעוד מצוין (CLAUDE.md, הערות בקוד)
- dedup מתוחכם שמטפל במגוון edge cases
- ניהול זיכרון מודע (חסימת משאבים, `gc.collect`, `about:blank`)
- fallback חכם בסיווג (batch → single, model rotation)
- ממשק ניהול מרובה ערוצים (טלגרם + פאנל ווב)

**מה דורש שיפור:**
- קריאות סינכרוניות שחוסמות event loop
- קובץ `main.py` ענק (1,670 שורות)
- אבטחת מידע רגיש
- חוסר rate limiting ו-retry logic
- חוסר מוניטורינג מובנה

---

## בעיות קוד ושיפורים

### 1. קריאות סינכרוניות ב-notifier.py חוסמות event loop (חומרה: גבוהה)

**הבעיה:** `notifier.py` משתמש ב-`requests.post()` סינכרוני. למרות שב-CLAUDE.md מתועד שקריאות סינכרוניות חייבות `asyncio.to_thread()` (באג #4), ה-wrapper נמצא רק ב-`main.py` — אבל `send_lead()` ב-`run_cycle()` (שורה 1471) נקרא ישירות בלי `asyncio.to_thread()`.

**קובץ:** `main.py:1471`, `notifier.py:36`

**הפתרון:** לעטוף את כל הקריאות ל-`send_lead` ול-`send_message` ב-`run_cycle` ב-`asyncio.to_thread()`, או להפוך את `notifier.py` ל-async עם `httpx` / `aiohttp`.

---

### 2. `classify_batch` סינכרוני ב-event loop (חומרה: גבוהה)

**הבעיה:** `classify_batch()` ב-`classifier.py` משתמש ב-`openai.OpenAI` (סינכרוני). הקריאה ב-`main.py:1535` (`results = classify_batch(to_classify)`) חוסמת את ה-event loop לאורך כל הסיווג.

**קובץ:** `main.py:1535`, `classifier.py:135`

**הפתרון:** להשתמש ב-`openai.AsyncOpenAI` או לעטוף ב-`asyncio.to_thread()`.

---

### 3. קובץ `main.py` ענק — 1,670 שורות (חומרה: בינונית)

**הבעיה:** הקובץ מכיל פונקציונליות מגוונת שלא קשורה אחת לשנייה:
- לוגיקת סינון (keyword filter, block, force_send, post age)
- ניהול מצב סריקה (scan_progress)
- לולאת טלגרם (~500 שורות של command handling)
- quiet hours / scheduling
- health server
- panel startup

זה מקשה על תחזוקה וקריאות.

**הפתרון:** לפצל ל-modules:
- `filters.py` — `passes_keyword_filter`, `is_blocked`, `matches_force_send`, `is_post_too_old`
- `telegram_bot.py` — `_telegram_control_loop` וכל ה-command handlers
- `scheduling.py` — quiet hours, interval logic

---

### 4. כפילות קוד בטיפול בפקודות טלגרם (חומרה: בינונית)

**הבעיה:** כל command handler ב-`_telegram_control_loop` עוקב אחרי אותו דפוס:
1. פירוס ארגומנט
2. ולידציה ("מילה ריקה" / "URL ריק")
3. קריאה לפונקציית DB
4. `reload_keywords()`
5. שליחת תשובה

הדפוס הזה חוזר ~15 פעמים (שורות 937-1091) עם הבדלים מינימליים.

**קובץ:** `main.py:937-1091`

**הפתרון:** ליצור helper שמטפל בדפוס:
```python
async def _handle_crud_command(text, chat_id, action_fn, reload_fn, usage_msg):
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await asyncio.to_thread(send_message, usage_msg, chat_id=chat_id)
        return
    ok, msg = action_fn(parts[1].strip())
    if ok and reload_fn:
        reload_fn()
    await asyncio.to_thread(send_message, msg, chat_id=chat_id)
```

---

### 5. thread-local SQLite connection ללא סגירה (חומרה: נמוכה-בינונית)

**הבעיה:** `_get_conn()` ב-`database.py` יוצר connection per thread אבל אף פעם לא סוגר אותם. ב-daemon threads (panel, health) ה-connections נשארים פתוחים עד שהתהליך מת.

**קובץ:** `database.py:28-31`

**הפתרון:** להוסיף `atexit` handler או context manager שסוגר connections. לחלופין, להשתמש ב-connection pool עם timeout.

---

### 6. Global mutable state מסוכן (חומרה: בינונית)

**הבעיה:** `BLOCK_KEYWORDS`, `PRE_FILTER_KEYWORDS`, `FORCE_SEND_KEYWORDS`, `GROUP_FORCE_SEND_KEYWORDS` הם globals שנטענים בזמן import ומתעדכנים דרך `reload_keywords()`. הפאנל (Flask thread) וה-main loop (asyncio thread) ניגשים אליהם בו-זמנית ללא locking.

ב-Python, הקריאה ל-list בדרך כלל thread-safe בגלל GIL, אבל `reload_keywords()` עושה assignment לכמה globals — אם thread אחד קורא `matches_force_send()` בזמן ש-thread אחר מריץ `reload_keywords()`, ייתכן מצב לא עקבי.

**קובץ:** `main.py:231-242`

**הפתרון:** לאחד את כל ה-globals למילון אחד עם `threading.Lock`, או להעביר ל-dataclass עם lock.

---

### 7. `_content_dedup_hash` מחושב מספר פעמים לאותו פוסט (חומרה: נמוכה)

**הבעיה:** ב-`run_cycle()`, ה-hash מחושב עד 3 פעמים לאותו פוסט:
1. שורה 1433 — בסינון כפילויות מוקדם
2. שורה 1466 — ב-force_send
3. שורה 1516 — לפני סיווג

**קובץ:** `main.py:1433, 1466, 1516`

**הפתרון:** לחשב פעם אחת ולשמור ב-`post["_content_hash"]` כבר בשלב הראשון.

---

### 8. `send_lead` לא מבצע retry (חומרה: בינונית)

**הבעיה:** אם שליחת הודעת טלגרם נכשלת (timeout, 5xx), הליד נשמר ב-DB אבל ההודעה לא נשלחה ולא תישלח שוב (כי `is_lead_sent` יחזיר `True` בסבב הבא).

**קובץ:** `main.py:1471-1479`, `notifier.py:36-49`

**הפתרון:** לבדוק את ערך ההחזרה של `send_lead()` ולשמור ב-DB רק אם ההודעה נשלחה בהצלחה. לחלופין, להוסיף retry עם backoff.

---

### 9. חסר timeout על SQLite queries (חומרה: נמוכה)

**הבעיה:** `sqlite3.connect(DB_PATH)` לא מגדיר timeout. ברירת המחדל היא 5 שניות, אבל ב-VM עם 512MB RAM, תחת לחץ I/O, ייתכן lock timeout.

**קובץ:** `database.py:30`

**הפתרון:** `sqlite3.connect(DB_PATH, timeout=10)` ו-WAL mode:
```python
conn.execute("PRAGMA journal_mode=WAL")
```

---

## בעיות אבטחה

### 10. credentials מאוחסנים ב-DB כ-plaintext (חומרה: גבוהה)

**הבעיה:** הפאנל שומר `fb_email` ו-`fb_password` ב-SQLite כטקסט פשוט בטבלת `_config`. כל מי שיש לו גישה לקובץ ה-DB יכול לקרוא את הסיסמה.

**קובץ:** `panel.py:139-143`, `database.py:308-314`

**הפתרון:** להצפין את הסיסמה עם `cryptography.fernet` לפני שמירה, או לא לשמור סיסמה בכלל ולהסתמך רק על env vars.

---

### 11. PANEL_TOKEN מושווה כ-string רגיל (חומרה: בינונית)

**הבעיה:** `auth == f"Bearer {panel_token}"` — השוואת string רגילה פגיעה ל-timing attack.

**קובץ:** `panel.py:63`

**הפתרון:** `hmac.compare_digest(auth, f"Bearer {panel_token}")`

---

### 12. הפאנל חושף סיסמה ב-GET response (חומרה: גבוהה)

**הבעיה:** `GET /api/settings` מחזיר את `fb_password` בתשובת ה-JSON. גם אם יש אימות, הסיסמה נחשפת ב-browser memory, network tab, ולוגים.

**קובץ:** `panel.py:124-133`

**הפתרון:** להחזיר `"fb_password": "****"` או `"fb_password_set": true/false` במקום הערך בפועל.

---

### 13. חסר rate limiting על API endpoints (חומרה: בינונית)

**הבעיה:** אין הגבלה על מספר בקשות לפאנל. תוקף שמגלה את ה-URL יכול לבצע brute-force על `PANEL_TOKEN` או להציף את השרת.

**הפתרון:** להוסיף `flask-limiter` עם הגבלה (למשל 30 בקשות/דקה), לפחות על `/api/auth`.

---

### 14. BOT_TOKEN נטען פעם אחת ב-module level (חומרה: נמוכה)

**הבעיה:** `BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")` נטען בזמן import. אם ה-token מתעדכן (rotation), צריך restart.

**קובץ:** `notifier.py:8`

**הפתרון:** לקרוא את ה-token בכל קריאה, או לכל הפחות לאפשר reload.

---

## שיפורי ביצועים וזיכרון

### 15. `_stable_text_for_hash` מריץ 8+ regex על כל שורה (חומרה: בינונית)

**הבעיה:** הפונקציה מריצה:
1. `_URL_RE.sub`
2. `_INVISIBLE_CHARS_RE.sub`
3. `_PUA_RE.sub`
4. `re.match(r'^\d[\d,. ]*$')` לכל שורה
5. `_EMOJI_RE.search` + `re.search(r'[a-zA-Zא-ת]')` לכל שורה
6. `_ENGAGEMENT_LINE_RE.match` לכל שורה
7. `_DYNAMIC_LINE_RE.match` לכל שורה
8. `_is_action_bar` לכל שורה
9. `_TIMESTAMP_RE.sub` על הטקסט המלא

עבור פוסטים ארוכים עם הרבה שורות, זה יכול להיות איטי.

**קובץ:** `scraper.py:208-257`

**הפתרון:** לאחד חלק מה-regexים לביטוי אחד (במיוחד הבדיקות line-level), או לבצע early return אם הטקסט קצר. לשקול caching עם `functools.lru_cache`.

---

### 16. `get_daily_stats` משתמש ב-LIKE על strings (חומרה: נמוכה)

**הבעיה:** `WHERE seen_at LIKE '2026-03-06%'` — LIKE על text field לא יעיל. ככל שהטבלה גדלה, השאילתא תהיה יותר איטית.

**קובץ:** `database.py:148-156`

**הפתרון:** להוסיף עמודת `date` נפרדת (או index על `substr(seen_at, 1, 10)`), או להשתמש ב-`BETWEEN`:
```sql
WHERE seen_at >= '2026-03-06' AND seen_at < '2026-03-07'
```

---

### 17. `cleanup_old_posts` ללא VACUUM (חומרה: נמוכה)

**הבעיה:** מחיקת רשומות ישנות לא מקטינה את גודל קובץ ה-DB. לאורך זמן (חודשים) הקובץ ימשיך לגדול.

**קובץ:** `database.py:401-408`

**הפתרון:** להריץ `VACUUM` מדי פעם (לא בכל סבב — זה יקר):
```python
if deleted > 1000:
    conn.execute("VACUUM")
```

---

## שיפורי ארכיטקטורה

### 18. חוסר הפרדה בין שכבת ה-data לשכבת ה-logic

**הבעיה:** `main.py` ניגש ישירות ל-`_get_conn()` של `database.py` (שורה 164-167 ב-`get_all_group_force_send`). זה שובר encapsulation ומקשה על שינוי שכבת ה-DB.

**קובץ:** `main.py:164`

**הפתרון:** להוסיף פונקציה ב-`database.py` שמחזירה את הנתונים, במקום גישה ישירה ל-connection.

---

### 19. חוסר structured logging

**הבעיה:** לוגים הם strings חופשיים. קשה לעשות parsing ולנתח אותם בכלי מוניטורינג.

**קובץ:** `logger.py`

**הפתרון:** להוסיף JSON logging (אופציונלי דרך env var):
```python
if os.environ.get("LOG_FORMAT") == "json":
    # JSON formatter
```

---

### 20. חסר מנגנון health check מעמיק

**הבעיה:** ה-health endpoint מחזיר תמיד `200 OK`, גם אם ה-DB לא נגיש, פייסבוק חסום, או הסריקה תקועה.

**קובץ:** `main.py:1163-1183`

**הפתרון:** health check שבודק:
- DB accessible
- זמן מאז סריקה אחרונה < threshold
- session file exists
- disk space

---

## הצעות לפיצ'רים חדשים

### פיצ'ר 1: דו"ח שבועי/חודשי אוטומטי

**תיאור:** שליחה אוטומטית של דו"ח שבועי לטלגרם: כמה לידים, מאיזה קבוצות, אחוז רלוונטיות, עלויות API.

**יתרון:** מאפשר ללקוח לראות ROI בלי לשאול.

**מורכבות:** נמוכה — הנתונים כבר ב-DB, צריך רק aggregation ותזמון.

**מימוש מוצע:**
- הוספת פקודה `/weekly_report`
- cron-like check בתחילת כל סבב: אם עבר שבוע מהדו"ח האחרון → שולח
- שמירת `last_weekly_report` ב-`_config`

---

### פיצ'ר 2: סיווג מדורג (scoring) במקום בינארי

**תיאור:** במקום `relevant: true/false`, להחזיר ציון 1-10 שמייצג רמת רלוונטיות. לשלוח רק ליד עם ציון מעל סף (הניתן להגדרה).

**יתרון:** מפחית false positives. מאפשר ללקוח לכוונן את הרגישות.

**מורכבות:** נמוכה — שינוי בפרומפט + שדה `score` בתוצאה.

**מימוש מוצע:**
```json
{"relevant": true, "score": 8, "reason": "..."}
```
- סף ברירת מחדל: 6 (ניתן לשנות דרך הפאנל)
- לשמור את הציון ב-`sent_leads` לניתוח עתידי

---

### פיצ'ר 3: ניתוח קבוצות — אילו קבוצות מייצרות הכי הרבה לידים

**תיאור:** dashboard שמראה לכל קבוצה: כמה פוסטים נסרקו, כמה לידים נשלחו, אחוז רלוונטיות.

**יתרון:** מאפשר ללקוח להסיר קבוצות לא פרודוקטיביות ולהתמקד בטובות.

**מורכבות:** נמוכה — הנתונים כבר ב-`sent_leads.group_name`.

**מימוש מוצע:**
```sql
SELECT group_name, COUNT(*) as leads
FROM sent_leads
GROUP BY group_name
ORDER BY leads DESC
```
- endpoint חדש: `GET /api/stats/groups`
- הצגה בפאנל ובפקודת טלגרם `/group_stats`

---

### פיצ'ר 4: תגובה אוטומטית לפוסט (auto-reply)

**תיאור:** כשנמצא ליד רלוונטי, הבוט יכול להגיב אוטומטית בפוסט עם הודעה מוגדרת מראש (למשל "שלחתי לך הודעה פרטית").

**יתרון:** תגובה מהירה = יתרון תחרותי. מגיע ללקוח לפני המתחרים.

**מורכבות:** בינונית — דורש Playwright action בזמן הסריקה. **סיכון:** יכול לגרום לחסימה בפייסבוק אם נעשה באגרסיביות.

**מימוש מוצע:**
- toggle דרך הפאנל/טלגרם (`auto_reply: on/off`)
- טקסט תגובה הניתן להגדרה
- rate limit: תגובה אחת ל-5 דקות
- הקפדה על cool-down בין תגובות

---

### פיצ'ר 5: התראות בזמן אמת עם priority levels

**תיאור:** ליד עם ציון גבוה (9-10) ישלח עם notification sound ב-טלגרם, ליד בינוני (6-8) ישלח שקט.

**יתרון:** הלקוח מקבל notification רק על לידים "חמים" — פחות רעש.

**מורכבות:** נמוכה — `disable_notification: true/false` ב-Telegram API.

---

### פיצ'ר 6: ייצוא לידים ל-CSV/Google Sheets

**תיאור:** פקודת `/export` שמייצרת קובץ CSV עם כל הלידים (תאריך, קבוצה, תוכן, סיבה, URL).

**יתרון:** מאפשר ללקוח לעקוב, לנהל CRM, לנתח trends.

**מורכבות:** נמוכה — `send_document` כבר קיים ב-`notifier.py`.

**מימוש מוצע:**
```python
import csv, io
buf = io.StringIO()
writer = csv.writer(buf)
writer.writerow(["date", "group", "content", "reason", "url"])
for lead in leads:
    writer.writerow([...])
send_document(buf.getvalue().encode('utf-8-sig'), "leads.csv")
```

---

### פיצ'ר 7: מנגנון feedback — "רלוונטי" / "לא רלוונטי" על כל ליד

**תיאור:** כל ליד שנשלח לטלגרם מגיע עם שני כפתורים: "רלוונטי" / "לא רלוונטי". הפידבק נשמר ב-DB ומשמש ל:
1. סטטיסטיקה (אחוז דיוק)
2. שיפור הפרומפט בעתיד

**יתרון:** מאפשר למדוד את איכות הסיווג ולשפר לאורך זמן.

**מורכבות:** בינונית — צריך `send_message_with_buttons` (כבר קיים), callback handler, ו-DB table.

**מימוש מוצע:**
- טבלה חדשה: `lead_feedback(post_id, feedback, created_at)`
- כפתורים: `[{"text": "👍", "callback_data": "fb:relevant:<id>"}, {"text": "👎", "callback_data": "fb:irrelevant:<id>"}]`
- פקודה `/accuracy` שמראה אחוז דיוק

---

### פיצ'ר 8: בדיקת בריאות סשן מתוזמנת

**תיאור:** כל X שעות, הבוט בודק שהסשן תקין (מבלי לסרוק קבוצות) — טוען דף פייסבוק בודד ומוודא שאין redirect ללוגין.

**יתרון:** גילוי מוקדם של סשן פגום — לפני שהסריקה נכשלת.

**מורכבות:** נמוכה — כבר יש `_has_login_overlay` ו-`_is_wui_page`.

---

### פיצ'ר 9: סריקה חכמה — תדירות דינמית לפי קבוצה

**תיאור:** קבוצות פעילות (הרבה פוסטים חדשים) נסרקות בתדירות גבוהה. קבוצות שקטות נסרקות פחות.

**יתרון:** חוסך זמן וזיכרון. מתמקד בקבוצות פרודוקטיביות.

**מורכבות:** בינונית — צריך לעקוב אחרי מספר פוסטים חדשים לכל קבוצה ב-DB.

---

### פיצ'ר 10: דו"ח עלויות חודשי ב-טלגרם

**תיאור:** שליחה אוטומטית של סיכום עלויות API חודשי. כבר יש `get_usage_stats` ו-`get_daily_usage_stats` — צריך רק aggregation חודשי ותזמון.

**יתרון:** שקיפות מלאה ללקוח על עלויות.

**מורכבות:** נמוכה.

---

## סיכום עדיפויות

### דחוף (לתקן עכשיו)
| # | נושא | סיבה |
|---|-------|-------|
| 10,12 | credentials ב-plaintext / חשיפה ב-API | אבטחה קריטית |
| 1,2 | קריאות סינכרוניות חוסמות event loop | יציבות — יכול לגרום ל-timeout ותקיעה |
| 8 | `send_lead` ללא retry / בדיקת הצלחה | איבוד לידים בשקט |

### חשוב (לתכנן בקרוב)
| # | נושא | סיבה |
|---|-------|-------|
| 3 | פיצול `main.py` | תחזוקתיות — הקובץ גדל ומקשה על שינויים |
| 6 | global mutable state | thread safety |
| 11 | timing-safe token comparison | אבטחה |
| 13 | rate limiting | אבטחה |

### רצוי (שיפור מתמשך)
| # | נושא | סיבה |
|---|-------|-------|
| 4 | כפילות קוד בטלגרם | DRY |
| 7 | חישוב hash כפול | ביצועים |
| 15 | אופטימיזציה ל-regex | ביצועים |
| 16,17 | DB optimizations | ביצועים לאורך זמן |

### פיצ'רים חדשים — לפי ROI
| עדיפות | פיצ'ר | סיבה |
|---------|--------|-------|
| 1 | פיצ'ר 7 (feedback) | מאפשר למדוד ולשפר — בסיס לכל השאר |
| 2 | פיצ'ר 6 (ייצוא CSV) | ערך ישיר ללקוח, מימוש קל |
| 3 | פיצ'ר 3 (ניתוח קבוצות) | אופטימיזציה — הכי הרבה ROI על הזמן |
| 4 | פיצ'ר 2 (scoring) | מפחית false positives |
| 5 | פיצ'ר 1 (דו"ח שבועי) | שקיפות ו-retention |
