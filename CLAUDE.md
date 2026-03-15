# CLAUDE.md — כללי עבודה לפרויקט Facebook Leads Bot

## סקירה
בוט שסורק קבוצות פייסבוק, מסנן לידים רלוונטיים באמצעות Claude API, ושולח אותם לטלגרם.
רץ על VM עם **512MB RAM** — כל שינוי חייב להתחשב במגבלת זיכרון.

## קבצים עיקריים

| קובץ | תפקיד |
|-------|--------|
| `scraper.py` | גלישה בפייסבוק עם Playwright (אתר מובייל) |
| `main.py` | אורקסטרציה, לולאת טלגרם, שעות שקטות |
| `classifier.py` | סיווג פוסטים עם OpenAI gpt-4.1-mini — תומך סיווג באצ'ים (`classify_batch`) + auto-fallback למודל הבא |
| `notifier.py` | שליחת הודעות טלגרם |
| `database.py` | SQLite — seen_posts, sent_leads, groups, keywords, _config |
| `logger.py` | לוגר |
| `tests.py` | טסטים עם pytest |

## באגים שתוקנו — לא לחזור עליהם

### 1. עצירה מוקדמת בגלל פוסט מוצמד (pinned)
**הבעיה:** פוסט מוצמד בראש הפיד כבר ב-DB → `seen_checker` מחזיר `True` → גלילה נעצרת מיד עם 0 פוסטים.
**הכלל:** תמיד לדרוש **רצף** של פוסטים מוכרים לפני עצירה (`KNOWN_THRESHOLD`), לא פוסט בודד.

### 2. מעקב לפי אינדקס שובר DOM וירטואלי
**הבעיה:** שימוש ב-`post_elements[index:]` נשבר כשפייסבוק מסיר/מוסיף אלמנטים בגלילה (virtualization).
**הכלל:** תמיד לעקוב לפי **תוכן** (`seen_texts` set), לא לפי אינדקס.

### 3. fallback URL יוצר post_id זהה לכל הפוסטים
**הבעיה:** כשחילוץ URL נכשל, ה-fallback הוא URL הקבוצה → כל הפוסטים מקבלים אותו ID → `seen_checker` עוצר הכל.
**הכלל:** לבדוק `seen_checker` רק כש-`has_real_url == True`.

### 4. חסימת event loop עם קריאות סינכרוניות
**הבעיה:** `requests.post()` חוסם את הלולאה האסינכרונית.
**הכלל:** כל קריאת רשת סינכרונית חייבת להיות עטופה ב-`asyncio.to_thread()`.

### 5. טלגרם 409 Conflict בין instances
**הבעיה:** שתי instances שעושות polling → 409 Conflict חוזר.
**הכלל:** backoff מעריכי על שגיאות 409.

### 6. בדיקת "טבלה מוגדרת" ברמת טבלה במקום ברמת סוג
**הבעיה:** `_table_was_configured("keywords")` בדקה ב-`sqlite_sequence` ברמת הטבלה. pre_filter ו-block חולקים את אותה טבלה, אז הוספת block סימנה את הטבלה כמוגדרת → `get_db_keywords("pre_filter")` החזיר `[]` במקום `None` → fallback ל-defaults אבד → הבוט הפסיק למצוא לידים.
**הכלל:** לעולם לא לבדוק configuration ברמת טבלה כשיש סוגים שונים באותה טבלה. להשתמש בטבלת `_config` עם מפתח **לכל סוג** (`keywords_pre_filter`, `keywords_block`, `groups`).

### 7. נרמול URL ללא סכמה שובר דומיינים מוכרים
**הבעיה:** `www.facebook.com/groups/123` בלי `https://` → ה-replace ל-`m.` רץ ראשון → בדיקת `http` נכשלת → prefix כפול: `https://m.facebook.com/groups/m.facebook.com/groups/123`.
**הכלל:** ב-`_normalize_group_url`, תמיד **להוסיף סכמה לפני** החלפת דומיין. לזכור לטפל גם ב-`facebook.com` חשוף (ללא `www`).

### 8. רשימת מילות מפתח ריקה חוסמת כל הפוסטים
**הבעיה:** `passes_keyword_filter` לא בדק רשימה ריקה. `any()` על `[]` מחזיר `False` → כל הפוסטים נדחים בשקט.
**הכלל:** פונקציות סינון חייבות guard על רשימה ריקה. הדפוס: **רשימה ריקה = אין סינון = מעביר הכל**. (ראו `is_blocked` כדוגמה נכונה.)

### 9. סיווג באצ'ים — תוצאות חלקיות גורמות ל-misalignment
**הבעיה:** ב-`classify_batch`, אם אלמנט בתשובת JSON לא תקין (למשל `null`), הלולאה שמוסיפה תוצאות ל-`all_results` זורקת exception אחרי הוספה חלקית. ה-fallback מוסיף את כל הבאצ' שוב → `all_results` ארוך מ-`posts` → `zip()` ב-`main.py` מתאים לידים לפוסטים לא נכונים.
**הכלל:** לאסוף תוצאות באצ' ברשימה זמנית (`batch_results`) ולהוסיף ל-`all_results` רק אחרי הצלחה מלאה.

### 10. זיהוי wui אגרסיבי מדי גורם ל-false positive ושובר סריקה
**הבעיה:** `_is_wui_page` בדק אם ≥50% מהלינקים הם `/wui/`. פייסבוק מובייל מוסיף כמה לינקי wui גם בדפים תקינים (באנר "פתח באפליקציה"), ובזמן רינדור הם מהווים רוב הלינקים → false positive → re-login → נכשל → 0 פוסטים מכל הקבוצות.
**הכלל:**
  - **סף מחמיר**: לפחות 3 לינקי wui + לפחות 75% מכלל הלינקים.
  - **בדיקה כפולה**: גם אם יחס wui גבוה, אם יש אלמנטי פוסטים בדף (`article`, `div[data-ft]`) — לא wui-only.
  - **כלל רוחב**: מנגנוני זיהוי סשן חייבים להיות שמרניים — false positive שובר את כל הסריקה, false negative רק מפספס סבב אחד.

### 11. חילוץ URL מפוסטים — מגבלה מובנית של m.facebook.com
**המצב:** `m.facebook.com` לא תמיד מכניס `<a href>` עם קישור לפוסט בודד ב-HTML. בחלק מהקבוצות/דפים כל הלינקים הם `/wui/action/` בלבד.
**הכלל:**
  - זו **לא שגיאה בקוד** — זו מגבלה של אתר המובייל. לא לנסות "לתקן" חילוץ URL כשאין לינקים ב-HTML.
  - ה-fallback (URL הקבוצה) הוא ההתנהגות הנכונה.
  - ה-`has_real_url` flag מבטיח ש-`seen_checker` לא ישתמש ב-fallback URL כ-post ID (ראו באג #3).

### 12. כפילויות בין סשנים — post_id לא יציב + חוסר dedup לפי תוכן
**הבעיה:** פוסטים נשלחו שוב בסשנים שונים בגלל שתי סיבות:
  1. **Hash לא יציב**: לפוסטים ללא URL אמיתי, ה-`post_id` מבוסס hash של `_stable_text_for_hash(text)`. אבל הפונקציה לא סיננה מטריקות engagement ("5 תגובות", "3 שיתופים") — כשהמספרים השתנו בין סריקות, ה-hash השתנה → פוסט זוהה כ"חדש".
  2. **אין dedup לפי תוכן ב-sent_leads**: גם כשאסטרטגיית חילוץ URL השתנתה (URL אמיתי בסשן אחד, fallback בסשן אחר), לא הייתה בדיקת תוכן שתתפוס כפילות.
**הכלל:**
  - `_stable_text_for_hash` חייב לסנן **גם** שורות engagement (תגובות, שיתופים, לייקים, צפיות) — דרך `_ENGAGEMENT_LINE_RE`. **זהירות עם אותיות סופיות**: ף ≠ פ (שיתוף vs שיתופים), לכן הרגקס משתמש ב-`[פף]`.
  - `sent_leads` כולל עמודת `content_hash` — hash מנורמל של תוכן הפוסט. לפני שליחת ליד, בודקים `is_content_hash_sent()` **בנוסף** ל-`is_lead_sent()`.
  - `save_lead()` שומר את ה-content_hash לצד ה-post_id.
  - **נרמול עמוק** ב-`_stable_text_for_hash` — lowercase, הסרת URLs (פרמטרי מעקב), הסרת תווי כיווניות/zero-width/PUA, הסרת מילות רעש UI (`_NOISE_WORDS`: "קרא עוד", "See more", "כתוב תגובה...", "ממומן" ועוד).
  - **דיאגנוסטיקה**: `[DEDUP-SEND]` מתעד כל ליד שנשלח (snippet + hash + post_id), `[DEDUP-STABLE]` מציג את הטקסט המנורמל המלא, `[DEDUP-CATCH]` מתעד כל כפילות שנתפסה לפי content_hash.

### 13. content_hash לא יציב — תגובות פייסבוק שמשנות את ה-hash
**הבעיה:** `innerText` של אלמנט הפוסט כולל גם תגובות/תשובות שמופיעות מתחת לפוסט. כשתגובות חדשות מתווספות בין סריקות, `_stable_text_for_hash` אמנם מסנן timestamps ו-engagement, אבל **לא יכול לסנן תוכן תגובות** (הוא טקסט רגיל). כתוצאה, ה-`content_hash` משתנה → כפילות לא נתפסת.
**הכלל:**
  - `_content_dedup_hash` משתמש רק ב-**12 המילים הראשונות** של הטקסט המנורמל ("core hash"). ליבת הפוסט (שם + תחילת הגוף) מופיעה תמיד בהתחלה ולא משתנה. תוכן דינמי (תגובות, engagement) מופיע בסוף ונחתך.
  - **מילים ולא תווים** — בפוסטים קצרים (~19 מילים / ~100 תווים) חיתוך לפי תווים (150) כולל חלק מתגובות. ספירת מילים לא מושפעת מאורך הפוסט.
  - `_NOISE_WORDS` כולל גם `"עוד"`, `"... עוד"`, `"הרלוונטיות ביותר"`, `"החדשות ביותר"`, `"most relevant"`, `"newest"`.
  - `_DYNAMIC_LINE_RE` מסנן שורות `"+N"` (ספירת ריאקשנים), `"ו-עוד N אחרים"`, `"and N others/more"`, וגם `"פלוני ו-N אחרים"`, `"John and N others"`.

### 14. כפילויות בין סשנים — תוכן דינמי ב-innerText שובר את ה-content_hash
**הבעיה:** פוסטים נשלחו שוב בסשנים שונים בגלל שלושה גורמים משולבים:
  1. **post_id לא עקבי**: חילוץ URL לא דטרמיניסטי (7 אסטרטגיות) — באחד הסשנים ה-URL חולץ, בשני לא → post_id שונה → `is_seen` ו-`is_lead_sent` לא תופסים.
  2. **תוכן דינמי בתחילת הטקסט**: שורות engagement עם אמוג'י (`👍 5`), תאריכים עבריים (`4 במרץ`), בר פעולות (`אהבתי · תגובה · שיתוף`) — לא סוננו ע"י `_stable_text_for_hash` → שינו את ה-content_hash.
  3. **בדיקת content_hash מאוחרת מדי**: content_hash נבדק רק אחרי סינון מילות מפתח — פוסטים שעברו `is_seen` עם post_id שונה לא נבדקו מספיק מוקדם.
**הכלל:**
  - **`_stable_text_for_hash` מסנן גם**: שורות אמוג'י ללא אותיות (`_EMOJI_RE`), engagement עם prefix אמוג'י (`👍 5 תגובות`), תאריכים עבריים (`4 במרץ`) ואנגליים (`March 4`), בר פעולות משולב (`_is_action_bar`), ורשימות מגיבים עם שמות (`"פלוני ו-3 אחרים"`).
  - **`_EMOJI_RE`**: מזהה אמוג'י סטנדרטיים (Emoticons, Pictographs, Transport, Flags ועוד). **חשוב**: לא לתפוס מספרי טלפון — דורשים נוכחות אמוג'י ולא רק "אין אותיות".
  - **`_ENGAGEMENT_LINE_RE`**: תומך ב-prefix אמוג'י אופציונלי לפני המספר (`^(?:[^\w\s]*\s*)?\d[...]`).
  - **`_TIMESTAMP_RE`**: תומך גם בתאריכים עבריים (`\d+ ב(ינואר|פברואר|מרץ|...)`) ואנגליים (`Jan/February/... \d+`), כולל `בשעה HH:MM` ו-`at HH:MM AM/PM`.
  - **בדיקת content_hash מוקדמת**: `[DEDUP-EARLY]` — מיד אחרי סינון `is_seen`, לפני סינון מילות מפתח. תופס כפילויות כש-post_id השתנה אבל התוכן זהה.
  - **אינדקס DB**: `idx_sent_leads_content_hash` על `sent_leads.content_hash` לביצועים.

### 15. לוגין במובייל נכשל בשרת חדש — mobile UA שובר form + security checkpoint
**הבעיה:** בשכפול ללקוח חדש (שרת חדש = IP חדש), הלוגין ב-m.facebook.com נכשל משתי סיבות:
  1. **אין form HTML אמיתי**: mobile UA (Pixel 7) → m.facebook.com → React, אין `<form>` → overlays "הורד אפליקציה" חוסמים → `form.submit()` / Enter נכשלים.
  2. **security checkpoint**: IP חדש + דפדפן headless → פייסבוק דורש אימות נוסף (`two_step_verification` / `checkpoint` / `approvals` ב-URL) גם בלי 2FA.
**הכלל:**
  - **לוגין תמיד בדסקטופ**: `login()` מקבל `browser` ויוצר context דסקטופ זמני (UA: Chrome/Windows, viewport: 1280x720) → `www.facebook.com/login/` → form HTML אמיתי עם email+password באותו דף.
  - **העברת cookies**: cookies של `.facebook.com` עובדים cross-subdomain → לוגין בדסקטופ, סריקה במובייל.
  - **context נסגר**: desktop context נסגר מיד אחרי העברת cookies (חסכון זיכרון).
  - **fallback**: אם `browser` לא מסופק, לוגין ישירות במובייל (תאימות לאחור).
  - **`_CHECKPOINT_MARKERS`**: `("checkpoint", "two_step_verification", "approvals")` — בדיקה ב-URL אחרי לוגין. הודעת שגיאה ברורה + צילום מסך `debug_checkpoint.png`.
  - **זה לא באג בקוד**: security checkpoint הוא החלטה של פייסבוק. הפתרון: אישור מכשיר ידני חד-פעמי (ראו `client_checklist.md` סעיף 2.8).

## כללי עבודה על session/auth detection
- **false positive שובר הכל**: זיהוי שגוי של "סשן פג" → re-login → נכשל → 0 פוסטים. לכן **סף גבוה + בדיקה כפולה**.
- **לפני שמניחים שפייסבוק שינה משהו** — לבדוק אם **הקוד עצמו השתנה** (רגרסיה). לפרוס גרסה ישנה ידועה כעובדת ולהשוות.
- **wui detection**: חייב גם יחס גבוה של לינקי wui (≥75%) **וגם** היעדר אלמנטי פוסטים. אחד לבד לא מספיק.
- **re-login flow**: Enter על שדה אימייל אמין יותר מחיפוש `button[type='submit']` (סלקטור גנרי שיכול לתפוס כפתור לא נכון).

## כללי scraping
- **אתר מובייל בלבד** (`m.facebook.com`) — חוסך ~60% זיכרון.
- **חסימת משאבים כבדים** — תמונות, וידאו, פונטים נחסמים ב-route handler.
- **`about:blank` בין קבוצות** — לשחרר DOM בזיכרון.
- **`gc.collect()`** אחרי סיום scraping ולפני סיווג.
- **גלילה חכמה** — לא מספר גלילות קבוע. גוללים עד רצף פוסטים מוכרים או `MAX_SCROLLS`.
- **חילוץ URL** — 7 אסטרטגיות (סלקטורים ישירים → סריקת לינקים → timestamp → data-attributes → element ID → DOM climbing → page-level scan). Fallback ל-URL קבוצה אם הכל נכשל. **שים לב:** `m.facebook.com` לא תמיד מכיל לינקים לפוסטים בודדים — ה-fallback הוא התנהגות צפויה, לא באג (ראו באג #11).

## סיווג (classifier.py)
- משתמש ב-OpenAI Chat Completions API (מודל `gpt-4.1-mini`). משתנה סביבה: `OPENAI_API_KEY`.
- `classify_post()` — סיווג פוסט בודד (fallback).
- `classify_batch()` — סיווג מספר פוסטים בבקשה אחת (ברירת מחדל: 5 לבאצ').
- `_chat_completion()` — פונקציית עזר לקריאת API עם auto-fallback למודל הבא אם הנוכחי deprecated (404/410).
- `_MODEL_PRIORITY` — רשימת מודלים לפי סדר עדיפות. כשמודל נכשל, עובר אוטומטית לבא ושומר ב-DB.
- פונקציית עזר `_parse_json_response()` — חילוץ JSON כולל unwrap של markdown. **חייבים להשתמש בה בכל פירוש תשובה מ-API** (לא לשכפל את הלוגיקה).
- fallback חכם: אם באצ' נכשל (שגיאת API, JSON לא תקין, אורך שגוי, אלמנט null) — נופל לסיווג בודד פוסט-פוסט.
- **פרומפט סיווג**: `_CLASSIFICATION_CRITERIA_DEFAULT` נטען מ-env var `CLASSIFICATION_CRITERIA`. המשתמש יכול להגדיר פרומפט מותאם דרך הפאנל (נשמר ב-DB כ-`classification_criteria`, עדיפות על env). אם שניהם ריקים — הבוט לא יסווג.

## async
- כל הקוד async. Playwright משתמש ב-async API.
- קריאות סינכרוניות (requests, sqlite) חייבות `asyncio.to_thread()`.
- task של טלגרם נשמר במשתנה (`_control_task`) כדי שלא ייאסף ע"י GC.

## טסטים
```bash
python -m pytest tests.py -v
```
339 טסטים ב-`tests.py` מכסים:
- `passes_keyword_filter()` — סינון לפי מילות מפתח (כולל רשימה ריקה)
- `is_blocked()` — חסימת מילים אסורות
- `extract_post_id()` — חילוץ ID מ-URLs בפורמטים שונים
- `_is_quiet_now()` — לוגיקת שעות שקטות (כולל מעבר חצות)
- `clean_post_content()` — ניקוי תוכן פוסטים
- `classify_post()` — סיווג בודד (עם מוקים)
- `classify_batch()` — סיווג באצ'ים (חלוקה, fallback, null elements, markdown)
- Groups CRUD — הוספה, מחיקה, נרמול URLs, fallback ל-None
- Keywords CRUD — הוספה, מחיקה, הפרדת סוגים, fallback ל-defaults
- `reload_keywords()` — טעינה מ-DB עם fallback

## התאמה ללקוח (`client_checklist.md`)
- קובץ `client_checklist.md` מפרט את כל המקומות בקוד שצריך לערוך כשמשכפלים את הפרויקט ללקוח חדש.
- **בכל שינוי בקובץ שנוגע בשורות/הגדרות המפורטות בצ'קליסט — לעדכן גם את `client_checklist.md`** (מספרי שורות, שמות משתנים, תבניות הודעות וכו').
- הקבצים הרלוונטיים: `classifier.py` (קריטריוני סיווג), `scraper.py` (קבוצות ברירת מחדל), `main.py` (מילות מפתח, הודעות טלגרם), `notifier.py` (תבנית הודעת ליד), `render.yaml` (ברירות מחדל deploy).

## שפה ונוהלי עבודה
- **תיאורי PR**: בעברית
- **הערות בקוד**: בעברית
- **commit messages**: באנגלית (conventional commits)

## פקודות הרצה
```bash
docker build -t fb-leads .
docker run --env-file .env fb-leads
```
