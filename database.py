import aiosqlite

DB_PATH = "bot.db"

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE,
                username TEXT,
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS posted_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                posted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS news_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                name TEXT,
                desc TEXT,
                pros TEXT,
                prompt TEXT,
                tool_name TEXT,
                tool_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()

async def add_user(user_id: int, username: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO users (user_id, username) VALUES (?, ?)",
            (user_id, username)
        )
        await db.commit()

async def get_all_users() -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT user_id FROM users") as cursor:
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

async def is_link_posted(url: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT 1 FROM posted_links WHERE url = ?", (url,)) as cursor:
            return await cursor.fetchone() is not None

async def save_link(url: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT OR IGNORE INTO posted_links (url) VALUES (?)", (url,))
        await db.commit()

async def save_post(url: str, name: str, desc: str, pros: list, prompt: str,
                    tool_name: str = "", tool_url: str = ""):
    import json
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR IGNORE INTO news_posts
               (url, name, desc, pros, prompt, tool_name, tool_url)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (url, name, desc, json.dumps(pros, ensure_ascii=False), prompt, tool_name, tool_url)
        )
        await db.commit()

async def search_news(query: str, limit: int = 5) -> list:
    import json
    pattern = f"%{query.lower()}%"
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """
            SELECT url, name, desc, pros, prompt, tool_name, tool_url
            FROM news_posts
            WHERE LOWER(name)      LIKE ?
               OR LOWER(desc)      LIKE ?
               OR LOWER(tool_name) LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (pattern, pattern, pattern, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "url":       row[0],
                    "name":      row[1],
                    "desc":      row[2],
                    "pros":      json.loads(row[3]),
                    "prompt":    row[4],
                    "tool_name": row[5] or "",
                    "tool_url":  row[6] or "",
                }
                for row in rows
            ]


async def get_top_tools(limit: int = 5) -> list:
    # deduplicated by tool_name, last 30 days
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """
            SELECT tool_name, tool_url, desc
            FROM news_posts
            WHERE created_at >= datetime('now', '-30 days')
              AND tool_name IS NOT NULL
              AND tool_name != ''
            GROUP BY tool_name
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {"tool_name": row[0], "tool_url": row[1] or "", "desc": row[2]}
                for row in rows
            ]


async def get_top_posts_month(limit: int = 7) -> list:
    # deprecated, use get_top_tools()
    import json
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """
            SELECT url, name, desc, pros, prompt
            FROM news_posts
            WHERE created_at >= datetime('now', '-30 days')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {"url": r[0], "name": r[1], "desc": r[2],
                 "pros": json.loads(r[3]), "prompt": r[4]}
                for r in rows
            ]

async def get_last_posts(limit: int = 3) -> list:
    import json
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT url, name, desc, pros, prompt FROM news_posts ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            result = []
            for row in rows:
                pros = json.loads(row[3])
                result.append({
                    "url":     row[0],
                    "name":    row[1],
                    "desc":    row[2],
                    "pros":    pros,
                    "feature": pros[0] if pros else "",  # first pro = main feature
                    "prompt":  row[4],
                })
            return result