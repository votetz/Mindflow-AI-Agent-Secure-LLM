# 💠 Mindflow AI — Autonomous Neural Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Aiogram-3.x-orange?style=for-the-badge&logo=telegram&logoColor=white" alt="Aiogram">
  <img src="https://img.shields.io/badge/Groq-LPU_Powered-green?style=for-the-badge&logo=nvidia&logoColor=white" alt="Groq">
</p>

**Mindflow AI** — це інтелектуальний автономний агент, побудований на базі моделі **Llama 3.3** та інфраструктури **Groq LPU**. Проєкт розроблений як приклад стійкої до відмов та безпечної архітектури для роботи з великими мовними моделями (LLM).

---

## 🚀 Основний функціонал

* **⚡ AI-Дайджест**: Автоматизований збір та аналіз новин через RSS-пайплайни.
* **🔎 Розумний пошук**: Система пошуку ШІ-інструментів за описом або категорією.
* **🪄 Prompt Trainer**: Оптимізація та покращення промптів за методиками Prompt Engineering.
* **🎙 Голосовий асистент**: Транскрибація мовлення в текст за допомогою **Whisper-large-v3**.
* **📊 Порівняння моделей**: Детальний аналіз та порівняння різних ШІ-рішень.

---

## 🛡 Архітектура безпеки

Проєкт впроваджує багаторівневу систему захисту для стабільної роботи в умовах лімітів API:

* **Logical Guard**: Вбудований захист від «математичного спаму» (тетрації, нескінченні степені $\pi$) через Regex-фільтрацію.
* **User-Based Quotas**: Індивідуальні ліміти на кількість символів (10k/год) та токенів, що запобігають блокуванню всього API-ключа через одного користувача.
* **Smart Throttling**: Локальне обмеження частоти запитів (2 сек) та автоматичний бан на 60 сек при виникненні помилки 429 (Rate Limit).
* **Windows Resilience**: Повна сумісність шляхів через `os.path.join` та автоматичне очищення тимчасових файлів у `finally` блоках.



---

## ⚙️ Технологічний стек

* **Framework**: `aiogram 3.x` (Asynchronous Telegram Framework).
* **AI Backend**: `Groq Cloud API` (Llama 3.3-70b-versatile).
* **Database**: `SQLite` з асинхронним доступом.
* **Automation**: `APScheduler` для регулярних завдань.

---

## 🛠 Швидке розгортання

1.  **Встановіть залежності**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Налаштуйте `.env`**:
    ```env
    BOT_TOKEN=your_telegram_token
    GROQ_API_KEY=your_groq_api_key
    ADMIN_ID=your_id
    ```

3.  **Запустіть бота**:
    ```bash
    python main.py
    ```

---