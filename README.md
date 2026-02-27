# hacaton Admin/admin test task

Проєкт для автоматичної генерації реалістичних діалогів служби підтримки та їх глибокого автоматизованого 
QA-аналізу. Використовує **Multi-LLM архітектуру** для досягнення максимальної ефективності та економії 
коштів: генерація даних делегована Google Gemini, а строгий логічний аналіз — моделям OpenAI.


 Інструкція із запуску (Локально через Docker)

Для запуску проєкту вам знадобиться лише встановлений [Docker](https://www.docker.com/).
1. Перейдіть в потрібну папку на локальному пристрої для подальшого копіювання репозиторію;

2. Клонування репозиторію

 `git clone <посилання-на-ваш-репозиторій>`
 
 `cd <назва-папки-проєкту>`

2. Створіть файл .env та вставте туди ВАШІ api ключі від openAI and gemini (приклад можете побачити в 
файлі .env.example

3. КОМАНДИ ДЛЯ ВІДРИТТЯ

docker build -t admin-hacaton-bot

для запуску генерацій датасетів:
docker run --rm -v "$(pwd):/Admin-admin-hacaton" --env-file .env admin-hacaton-bot python generate.py

для запуску аналізу:
docker run --rm -v "$(pwd):/Admin-admin-hacaton" --env-file .env admin-hacaton-bot python analyze.py
