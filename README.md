# MedCenter AI Agent

Виртуальный помощник для медицинского центра, который может отвечать на голосовые запросы пациентов, предоставлять информацию об услугах и помогать с записью на прием.

## Возможности

- Распознавание речи с микрофона (Yandex SpeechKit)
- Синтез речи (Yandex SpeechKit)
- Обработка запросов с помощью LLM (Groq)
- Многоязычная поддержка (русский, узбекский)
- Распознавание намерений пользователя
- Помощь в записи на прием
- Предоставление информации об услугах медицинского центра
- Перенаправление экстренных запросов

## Требования

- Python 3.8+ 
- FFmpeg
- Microphone
- API-ключи:
  - Yandex SpeechKit API Key
  - Groq API Key

## Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/yourusername/medcenter-ai-agent.git
cd medcenter-ai-agent
```

2. Создайте и активируйте виртуальную среду:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Установите зависимости:

```bash
pip install -r requirements.txt
```

4. Установите FFmpeg (если еще не установлен):

- **Windows**: Загрузите с [ffmpeg.org](https://ffmpeg.org/download.html) и добавьте в PATH
- **Mac**: `brew install ffmpeg`
- **Linux**: `apt-get install ffmpeg`

5. Создайте файл `.env` с API-ключами:

```
YANDEX_API_KEY=your_yandex_api_key
GROQ_API_KEY=your_groq_api_key
FFMPEG_PATH=/path/to/ffmpeg
FFPROBE_PATH=/path/to/ffprobe
```

## Использование

### Запуск в интерактивном текстовом режиме

```bash
python main.py --interactive
```

### Запуск с непрерывным прослушиванием микрофона

```bash
python main.py --continuous
```

### Запуск для одиночного цикла разговора

```bash
python main.py
```

### Дополнительные параметры

```
--yandex-api-key KEY  : Yandex SpeechKit API key
--groq-api-key KEY    : Groq API key
--voice VOICE         : Голос для синтеза речи (по умолчанию: yulduz_ru)
--voice-role ROLE     : Роль голоса (по умолчанию: friendly)
--voice-speed SPEED   : Скорость речи (по умолчанию: 1.1)
--log-level LEVEL     : Уровень логирования (по умолчанию: INFO)
--log-file FILE       : Файл логов
```

## Структура проекта

```
medcenter_ai/
├── config/
│   ├── __init__.py
│   └── settings.py          # Настройки конфигурации
├── services/
│   ├── __init__.py
│   ├── stt_service.py       # Сервис распознавания речи
│   ├── tts_service.py       # Сервис синтеза речи
│   ├── llm_service.py       # Интеграция с LLM (Groq)
│   └── audio_service.py     # Сервис обработки аудио
├── agents/
│   ├── __init__.py
│   ├── base_agent.py        # Базовый класс агента
│   └── medical_agent.py     # Агент медицинского центра
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py       # Утилиты для работы с аудио
│   └── logging_utils.py     # Утилиты для логирования
├── data/
│   ├── prompts/             # Промпты для LLM
│   │   └── medical_prompts.py
│   └── voices/              # Конфигурация голосов (если есть)
├── tests/
│   ├── __init__.py
│   ├── test_stt.py
│   ├── test_tts.py
│   └── test_agent.py
├── requirements.txt         # Зависимости проекта
├── main.py                  # Основная точка входа
└── README.md                # Документация проекта
```

## Расширение функциональности

### Добавление новых интентов

Для добавления новых интентов:

1. Обновите метод `_detect_intent` в `agents/medical_agent.py`
2. Добавьте соответствующие обработчики и промпты в `data/prompts/medical_prompts.py`

### Интеграция с CRM/EHR системами

Для интеграции с системами управления медицинским центром:

1. Создайте новый сервис в `services/` (например, `crm_service.py`)
2. Добавьте методы для взаимодействия с API CRM-системы
3. Зарегистрируйте сервис в агенте через `register_service`

## Лицензия

MIT

## Авторы

Ваше имя и контактная информация
