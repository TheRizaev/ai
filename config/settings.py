"""
Упрощенная конфигурация для простого голосового агента.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Yandex Cloud API settings
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY', '')
YANDEX_STT_ENDPOINT = 'stt.api.cloud.yandex.net:443'
YANDEX_TTS_ENDPOINT = 'tts.api.cloud.yandex.net:443'

# OpenAI API settings  
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = 'gpt-4o'  # Современная модель для разговоров

# Audio settings
CHUNK_SIZE = 4000
FORMAT = 'int16'  # Will be translated to pyaudio.paInt16 in the service
CHANNELS = 1
RATE = 8000

# FFmpeg paths
if os.name == 'nt':  # Windows
    FFMPEG_PATH = os.getenv('FFMPEG_PATH', os.path.join(BASE_DIR, 'ffmpeg', 'bin', 'ffmpeg.exe'))
    FFPROBE_PATH = os.getenv('FFPROBE_PATH', os.path.join(BASE_DIR, 'ffmpeg', 'bin', 'ffprobe.exe'))
else:  # Linux/Mac
    FFMPEG_PATH = os.getenv('FFMPEG_PATH', os.path.join(BASE_DIR, 'ffmpeg', 'bin', 'ffmpeg'))
    FFPROBE_PATH = os.getenv('FFPROBE_PATH', os.path.join(BASE_DIR, 'ffmpeg', 'bin', 'ffprobe'))

# Voice settings для дружелюбного разговора
VOICE = 'yulduz_ru'  # Женский русский голос
VOICE_ROLE = 'friendly'  # Дружелюбная интонация
VOICE_SPEED = 1.0  # Нормальная скорость для разговора

# Language settings
LANGUAGES = ['ru-RU', 'uz-UZ']  # Поддерживаемые языки

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'voice_agent.log')

# Agent settings для простого разговора
AGENT_NAME = 'Марина'
DEFAULT_SYSTEM_PROMPT = """Ты дружелюбный собеседник по имени {name}. 
Веди естественный разговор на русском языке. 
Отвечай кратко и по существу, 1-2 предложения максимум.
Будь живой, эмоциональной и интересной собеседницей.
Можешь задавать встречные вопросы для поддержания диалога, но не всегда."""