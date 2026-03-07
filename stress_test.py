import asyncio
import time
import logging

# Імітуємо твій обробник з services.py для перевірки логіки
from services import ask_groq, user_contexts, _429_bans

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


async def simulate_user(user_id: int, messages_count: int, delay: float, text: str):
    """
    Імітує поведінку окремого користувача.
    """
    logging.info(f"🚀 Користувач {user_id} почав активність")

    for i in range(messages_count):
        logging.info(f"👤 User {user_id} надсилає запит {i + 1}...")

        # Викликаємо твою основну функцію обробки
        response = await ask_groq(user_id, text)

        logging.info(f"🤖 Відповідь для User {user_id}: {response[:50]}...")

        if "⚠️" in response:
            logging.warning(f"❗ Користувач {user_id} отримав ОБМЕЖЕННЯ")

        await asyncio.sleep(delay)


async def run_stress_test():
    print("--- ЗАПУСК СТРЕС-ТЕСТУ Mindflow AI ---")

    # Сценарій:
    # 1. 'Шкідник' (User 1) - шле багато символів дуже часто
    # 2. 'Звичайний юзер' (User 2) - пише раз на 10 секунд

    tasks = [
        # User 1: 5 швидких запитів, що мають викликати бан
        simulate_user(user_id=111, messages_count=5, delay=0.5, text="A" * 60),

        # User 2: Спокійний запит, який має пройти, поки перший у бані
        simulate_user(user_id=222, messages_count=2, delay=10.0, text="Привіт, як справи?")
    ]

    await asyncio.gather(*tasks)
    print("--- ТЕСТ ЗАВЕРШЕНО ---")


if __name__ == "__main__":
    asyncio.run(run_stress_test())