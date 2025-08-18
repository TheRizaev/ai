"""
Новый главный файл для простого голосового агента.
Заменяет старый сложный main.py
"""
import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional
# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from config.settings import (
    YANDEX_API_KEY, OPENAI_API_KEY, 
    VOICE, VOICE_ROLE, VOICE_SPEED,
    LOG_LEVEL, LOG_FILE
)
from utils.logging_utils import setup_logging
from services.stt_service import STTService
from services.tts_service import TTSService
from services.llm_service import LangChainLLMService
from utils.audio_utils import AudioPlayer

logger = logging.getLogger(__name__)

class SimpleVoiceAgent:
    """Простой голосовой агент для разговоров с поддержкой LangChain."""
    
    def __init__(self, name="Марина"):
        """Инициализация агента."""
        self.name = name
        self.conversation_history = []
        self.max_history = 6  # Помним только последние 3 пары реплик
        self.current_chain = None  # Текущая активная цепочка
        
        # Сервисы будут подключены отдельно
        self.stt_service = None
        self.tts_service = None 
        self.llm_service = None
        
        logger.info(f"Создан простой голосовой агент '{name}' с поддержкой LangChain")
    
    def connect_services(self, stt_service, tts_service, llm_service):
        """Подключение сервисов."""
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.llm_service = llm_service
        
        # Создаем несколько готовых цепочек для демонстрации
        self._setup_demo_chains()
        
        logger.info("Сервисы подключены к агенту")
    
    def _setup_demo_chains(self):
        """Настройка демонстрационных цепочек."""
        if not self.llm_service:
            return
            
        try:
            # 1. Цепочка для анекдотов и юмора
            humor_prompt = """Ты {agent_name} - веселая собеседница, которая любит анекдоты и юмор.
            Отвечай с юмором, рассказывай анекдоты, шути.
            Всегда отвечай кратко - максимум 1-2 предложения.
            Можешь предлагать новые анекдоты."""
            
            self.llm_service.create_custom_chain(
                chain_name="humor",
                system_prompt=humor_prompt,
                temperature=0.9,
                max_tokens=120
            )
            
            # 2. Ограниченная цепочка только для разговоров о погоде
            self.llm_service.add_constraint_chain(
                chain_name="weather_only",
                allowed_topics=["погода", "температура", "дождь", "снег", "солнце", "облака"],
                forbidden_topics=["политика", "медицина", "финансы"]
            )
            
            # 3. Серьезная цепочка для деловых разговоров
            business_prompt = """Ты {agent_name} - профессиональный помощник для деловых вопросов.
            Говори серьезно, по делу, без шуток.
            Отвечай конкретно и информативно.
            Максимум 2 предложения."""
            
            self.llm_service.create_custom_chain(
                chain_name="business",
                system_prompt=business_prompt,
                temperature=0.3,
                max_tokens=100
            )
            
            logger.info("Демонстрационные цепочки настроены")
            
        except Exception as e:
            logger.error(f"Ошибка настройки цепочек: {e}")
    
    def switch_chain(self, chain_name: Optional[str] = None):
        """Переключение между цепочками."""
        available_chains = ["default", "humor", "weather_only", "business"]
        
        if chain_name and chain_name in available_chains:
            self.current_chain = chain_name if chain_name != "default" else None
            logger.info(f"Переключено на цепочку: {chain_name}")
            return f"Переключаюсь в режим '{chain_name}'. Теперь я буду отвечать по-другому!"
        else:
            chains_info = ", ".join(available_chains)
            return f"Доступные режимы: {chains_info}. Скажите 'режим [название]' для переключения."
    
    def handle_chain_commands(self, user_message: str) -> Optional[str]:
        """Обработка команд переключения режимов."""
        user_lower = user_message.lower()
        
        # Команды переключения режимов
        if "режим" in user_lower or "переключ" in user_lower:
            if "юмор" in user_lower or "анекдот" in user_lower or "humor" in user_lower:
                return self.switch_chain("humor")
            elif "погода" in user_lower or "weather" in user_lower:
                return self.switch_chain("weather_only")
            elif "деловой" in user_lower or "бизнес" in user_lower or "business" in user_lower:
                return self.switch_chain("business")
            elif "обычный" in user_lower or "default" in user_lower or "стандарт" in user_lower:
                return self.switch_chain("default")
            else:
                return self.switch_chain(None)  # Покажет доступные режимы
        
        # Команда для показа текущего режима
        if "какой режим" in user_lower or "текущий режим" in user_lower:
            current = self.current_chain or "default"
            return f"Сейчас активен режим: {current}"
        
        return None
    
    def add_to_history(self, role, message):
        """Добавление в историю разговора."""
        self.conversation_history.append({"role": role, "content": message})
        
        # Ограничиваем историю
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def listen(self):
        """Слушать пользователя."""
        if not self.stt_service:
            logger.error("STT сервис не подключен")
            return None
            
        print("🎤 Слушаю... (начните говорить)")
        
        try:
            # Коллбэк для отображения статуса
            def status_callback(event_type, data=None):
                if event_type == "start_listening":
                    print("🟢 Запись началась...")
                elif event_type == "partial" and data:
                    print(f"⚡ {data}")  # Показываем частичные результаты
                elif event_type == "recognized" and data:
                    print(f"✅ Распознано: {data}")
                elif event_type == "stop_listening":
                    print("🔴 Запись остановлена")
                elif event_type == "error":
                    print(f"❌ Ошибка: {data}")
            
            recognized_text = self.stt_service.recognize_stream(callback=status_callback)
            
            if recognized_text and recognized_text.strip():
                return recognized_text.strip()
            else:
                print("❌ Ничего не распознано")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка распознавания: {e}")
            print(f"❌ Ошибка при распознавании речи")
            return None
    
    def think(self, user_message):
        """Обдумать ответ."""
        if not self.llm_service:
            logger.error("LLM сервис не подключен")
            return "Извините, у меня проблемы с мышлением."
        
        print("🤔 Думаю...")
        
        # Проверяем команды переключения режимов
        chain_response = self.handle_chain_commands(user_message)
        if chain_response:
            return chain_response
        
        # Добавляем сообщение пользователя в историю
        self.add_to_history("user", user_message)
        
        try:
            # Используем LangChain сервис с текущей цепочкой
            if self.current_chain:
                print(f"🔗 Использую режим: {self.current_chain}")
                response = self.llm_service.generate_response(
                    user_input=user_message,
                    agent_name=self.name,
                    chain_name=self.current_chain
                )
            else:
                # Используем стандартную цепочку с историей
                response = self.llm_service.generate_with_history(
                    user_input=user_message,
                    conversation_history=self.conversation_history[:-1],
                    agent_name=self.name
                )
            
            if response:
                # Добавляем ответ в историю
                self.add_to_history("assistant", response)
                return response
            else:
                return "Хм, что-то я задумалась... Повторите, пожалуйста?"
                
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return "Извините, что-то с моими мыслями не так."
    
    def speak(self, message):
        """Произнести ответ."""
        if not self.tts_service:
            logger.error("TTS сервис не подключен")
            print(f"💬 {self.name}: {message}")
            return
        
        print(f"💬 {self.name}: {message}")
        print("🔊 Говорю...")
        
        try:
            # Синтезируем речь
            audio = self.tts_service.synthesize(
                text=message,
                voice=VOICE,
                role=VOICE_ROLE, 
                speed=VOICE_SPEED
            )
            
            if audio:
                AudioPlayer.play_audio_segment(audio)
                print("✅ Сказала")
            else:
                print("❌ Не смогла произнести")
                
        except Exception as e:
            logger.error(f"Ошибка синтеза речи: {e}")
            print("❌ Ошибка при произношении")
    
    def start_conversation(self):
        """Начать разговор."""
        print(f"\n🎉 Привет! Меня зовут {self.name}!")
        print("🗣️  Давайте просто поговорим. Скажите что-нибудь!")
        print("🔗 Новинка: Теперь я умею переключать режимы!")
        print("   • Скажите 'режим юмор' - для анекдотов")
        print("   • Скажите 'режим погода' - только о погоде")  
        print("   • Скажите 'режим деловой' - для серьезных тем")
        print("   • Скажите 'режим обычный' - стандартный режим")
        print("💡 Для выхода скажите 'пока', 'до свидания' или нажмите Ctrl+C\n")
        
        # Приветственное сообщение голосом
        welcome_message = f"Привет! Меня зовут {self.name}. Теперь у меня есть разные режимы общения! Попробуйте сказать 'режим юмор' или просто поговорим!"
        self.speak(welcome_message)
        
        conversation_count = 0
        
        try:
            while True:
                print(f"\n--- Реплика {conversation_count + 1} ---")
                
                # 1. Слушаем пользователя
                user_message = self.listen()
                
                if not user_message:
                    print("🤷 Попробуйте еще раз...")
                    continue
                
                # Проверяем на команды выхода
                if any(word in user_message.lower() for word in 
                       ['пока', 'до свидания', 'прощай', 'выход', 'хватит', 'стоп']):
                    farewell = "До свидания! Было приятно поговорить!"
                    self.speak(farewell)
                    break
                
                # 2. Думаем над ответом
                response = self.think(user_message)
                
                # 3. Отвечаем голосом
                self.speak(response)
                
                conversation_count += 1
                
        except KeyboardInterrupt:
            print("\n\n👋 Прощание...")
            farewell = "Пока! Удачи!"
            self.speak(farewell)
        except Exception as e:
            logger.error(f"Ошибка в разговоре: {e}")
            print(f"❌ Что-то пошло не так: {e}")


def setup_parser():
    """Настройка аргументов командной строки."""
    parser = argparse.ArgumentParser(description='Простой голосовой агент')
    
    parser.add_argument(
        '--yandex-api-key', 
        help='Yandex SpeechKit API key'
    )
    parser.add_argument(
        '--openai-api-key', 
        help='OpenAI API key'
    )
    parser.add_argument(
        '--name',
        default='Марина',
        help='Имя агента (по умолчанию: Марина)'
    )
    parser.add_argument(
        '--log-level', 
        default=LOG_LEVEL,
        help=f'Уровень логирования (по умолчанию: {LOG_LEVEL})'
    )
    
    return parser


def main():
    """Главная функция."""
    # Парсим аргументы
    parser = setup_parser()
    args = parser.parse_args()
    
    # Настраиваем логирование
    setup_logging(args.log_level, LOG_FILE)
    logger.info("🚀 Запуск простого голосового агента")
    
    # Получаем API ключи
    yandex_api_key = args.yandex_api_key or YANDEX_API_KEY or os.getenv('YANDEX_API_KEY')
    openai_api_key = args.openai_api_key or OPENAI_API_KEY or os.getenv('OPENAI_API_KEY')
    
    if not yandex_api_key:
        print("❌ Ошибка: Не указан Yandex API ключ")
        return 1
        
    if not openai_api_key:
        print("❌ Ошибка: Не указан OpenAI API ключ")
        return 1
    
    try:
        print("🔧 Инициализация сервисов...")
        
        # Создаем сервисы
        stt_service = STTService(api_key=yandex_api_key)
        tts_service = TTSService(api_key=yandex_api_key)
        llm_service = LangChainLLMService(api_key=openai_api_key)
        
        print("✅ Сервисы созданы")
        
        # Создаем и настраиваем агента
        agent = SimpleVoiceAgent(name=args.name)
        agent.connect_services(stt_service, tts_service, llm_service)
        
        print("✅ Агент готов к работе")
        
        # Начинаем разговор
        agent.start_conversation()
        
    except KeyboardInterrupt:
        print("\n👋 Программа завершена пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        print(f"💥 Критическая ошибка: {e}")
        return 1
    finally:
        # Закрываем сервисы
        try:
            if 'stt_service' in locals():
                stt_service.close()
            if 'tts_service' in locals():
                tts_service.close()
        except:
            pass
            
    logger.info("👋 Работа агента завершена")
    return 0


if __name__ == "__main__":
    sys.exit(main())