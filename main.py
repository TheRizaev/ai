"""
Обновленный главный файл для медицинского AI агента с RAG.
"""
import os
import sys
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
from utils.logging_utils import setup_logging, ConversationLogger
from services.stt_service import STTService
from services.tts_service import TTSService
from services.llm_service import LangChainLLMService
from services.rag_service import RAGService
from services.medical_db_service import MedicalDBService
from services.appointment_service import AppointmentService
from agents.medical_agent import MedicalAgent
from utils.audio_utils import AudioPlayer

logger = logging.getLogger(__name__)

class MedicalVoiceAssistant:
    """Медицинский голосовой помощник с поддержкой RAG."""
    
    def __init__(self, agent_name: str = "Марина", medical_center: str = "Таблетка"):
        """
        Инициализация медицинского ассистента.
        
        Args:
            agent_name: Имя агента
            medical_center: Название медицинского центра
        """
        self.agent_name = agent_name
        self.medical_center = medical_center
        
        # Сервисы
        self.stt_service = None
        self.tts_service = None
        self.llm_service = None
        self.rag_service = None
        self.medical_db_service = None
        self.appointment_service = None
        
        # Медицинский агент
        self.medical_agent = None
        
        # Логгер разговоров
        self.conversation_logger = None
        
        logger.info(f"Создан медицинский голосовой ассистент '{agent_name}' для центра '{medical_center}'")
    
    def initialize_services(self, yandex_api_key: str, openai_api_key: str):
        """
        Инициализация всех сервисов.
        
        Args:
            yandex_api_key: API ключ Yandex SpeechKit
            openai_api_key: API ключ OpenAI
        """
        try:
            print("🔧 Инициализация сервисов...")
            
            # 1. Базовые сервисы (STT, TTS, LLM)
            print("  📢 Настройка распознавания речи...")
            self.stt_service = STTService(api_key=yandex_api_key)
            
            print("  🔊 Настройка синтеза речи...")
            self.tts_service = TTSService(api_key=yandex_api_key)
            
            print("  🧠 Настройка языковой модели...")
            self.llm_service = LangChainLLMService(api_key=openai_api_key)
            
            # 2. RAG сервис
            print("  📚 Настройка базы знаний (RAG)...")
            self.rag_service = RAGService(openai_api_key=openai_api_key)
            
            # 3. Медицинская база данных
            print("  🏥 Настройка медицинской базы данных...")
            self.medical_db_service = MedicalDBService()
            
            # 4. Сервис записи на прием
            print("  📅 Настройка сервиса записи...")
            self.appointment_service = AppointmentService(self.medical_db_service)
            
            # 5. Медицинский агент
            print("  👩‍⚕️ Создание медицинского агента...")
            self.medical_agent = MedicalAgent(
                name=self.agent_name,
                medical_center_name=self.medical_center
            )
            
            # Подключаем сервисы к агенту
            self.medical_agent.connect_services(
                stt_service=self.stt_service,
                tts_service=self.tts_service,
                llm_service=self.llm_service
            )
            
            # Регистрируем медицинские сервисы
            self.medical_agent.register_services(
                rag_service=self.rag_service,
                medical_db_service=self.medical_db_service,
                appointment_service=self.appointment_service
            )
            
            # 6. Логгер разговоров
            print("  📝 Настройка логирования разговоров...")
            self.conversation_logger = ConversationLogger()
            
            print("✅ Все сервисы успешно инициализированы!")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации сервисов: {e}")
            raise
    
    def listen_to_user(self) -> Optional[str]:
        """Слушать пользователя."""
        try:
            print("\n🎤 Слушаю вас... (начните говорить)")
            
            def status_callback(event_type, data=None):
                if event_type == "start_listening":
                    print("🟢 Запись началась...")
                elif event_type == "partial" and data:
                    print(f"⚡ {data}")
                elif event_type == "recognized" and data:
                    print(f"✅ Распознано: {data}")
                elif event_type == "stop_listening":
                    print("🔴 Запись остановлена")
                elif event_type == "error":
                    print(f"❌ Ошибка: {data}")
            
            user_input = self.stt_service.recognize_stream(callback=status_callback)
            
            if user_input and user_input.strip():
                # Логируем ввод пользователя
                if self.conversation_logger:
                    self.conversation_logger.log_user_input(user_input)
                
                return user_input.strip()
            else:
                print("❌ Ничего не распознано")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка распознавания речи: {e}")
            print("❌ Ошибка при распознавании речи")
            return None
    
    def generate_response(self, user_message: str) -> str:
        """Генерация ответа через медицинского агента."""
        try:
            print("🤔 Анализирую запрос...")
            
            # Логируем системное событие
            if self.conversation_logger:
                self.conversation_logger.log_system_event("Обработка запроса", user_message[:50])
            
            # Используем медицинского агента для генерации ответа
            response = self.medical_agent.generate_response(user_message)
            
            if response:
                # Логируем ответ агента
                if self.conversation_logger:
                    self.conversation_logger.log_agent_response(response)
                
                return response
            else:
                return "Извините, что-то пошло не так. Попробуйте еще раз."
                
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return "Извините, произошла техническая ошибка. Попробуйте переформулировать ваш запрос."
    
    def speak_response(self, message: str):
        """Произнести ответ."""
        try:
            print(f"\n💬 {self.agent_name}: {message}")
            print("🔊 Произношу ответ...")
            
            # Синтезируем речь
            audio = self.tts_service.synthesize(
                text=message,
                voice=VOICE,
                role=VOICE_ROLE,
                speed=VOICE_SPEED
            )
            
            if audio:
                AudioPlayer.play_audio_segment(audio)
                print("✅ Ответ произнесен")
            else:
                print("❌ Не удалось произнести ответ")
                
        except Exception as e:
            logger.error(f"Ошибка синтеза речи: {e}")
            print("❌ Ошибка при произношении ответа")
    
    def start_medical_conversation(self):
        """Начать медицинскую консультацию."""
        try:
            print("\n" + "="*60)
            print(f"🏥 МЕДИЦИНСКИЙ ЦЕНТР '{self.medical_center.upper()}'")
            print(f"👩‍⚕️ Ваш помощник: {self.agent_name}")
            print("="*60)
            
            print("\n🌟 Я помогу вам:")
            print("   📅 Записаться на прием к врачу")
            print("   💰 Узнать стоимость услуг")
            print("   ℹ️  Получить информацию о специалистах")
            print("   ⏰ Узнать режим работы")
            print("   🩺 Ответить на общие медицинские вопросы")
            
            print("\n⚠️  ВАЖНО: Я НЕ заменяю консультацию врача!")
            print("💡 Для выхода скажите 'до свидания' или нажмите Ctrl+C")
            
            # Приветственное сообщение голосом
            welcome_message = (f"Добро пожаловать в медицинский центр {self.medical_center}! "
                             f"Меня зовут {self.agent_name}. Я помогу записаться на прием "
                             f"и отвечу на ваши вопросы. Чем могу помочь?")
            
            self.speak_response(welcome_message)
            
            conversation_count = 0
            
            while True:
                print(f"\n" + "-"*40 + f" Диалог {conversation_count + 1} " + "-"*40)
                
                # 1. Слушаем пользователя
                user_message = self.listen_to_user()
                
                if not user_message:
                    print("🤷 Попробуйте еще раз...")
                    continue
                
                # Проверяем команды выхода
                if any(word in user_message.lower() for word in 
                       ['пока', 'до свидания', 'прощай', 'выход', 'хватит', 'стоп', 'спасибо за помощь']):
                    farewell = f"До свидания! Берегите здоровье и обращайтесь в медицинский центр {self.medical_center}, если понадобится помощь!"
                    self.speak_response(farewell)
                    break
                
                # 2. Генерируем ответ
                response = self.generate_response(user_message)
                
                # 3. Произносим ответ
                self.speak_response(response)
                
                conversation_count += 1
                
                # Проверяем, не слишком ли длинный разговор
                if conversation_count >= 20:
                    reminder = ("Мы уже долго разговариваем. Если у вас есть еще вопросы, "
                               "обращайтесь в любое время. Берегите здоровье!")
                    self.speak_response(reminder)
                    break
                
        except KeyboardInterrupt:
            print("\n\n👋 Завершение работы...")
            farewell = "До свидания! Будьте здоровы!"
            self.speak_response(farewell)
        except Exception as e:
            logger.error(f"Ошибка в медицинском разговоре: {e}")
            print(f"💥 Произошла ошибка: {e}")
    
    def run_text_mode(self):
        """Запуск в текстовом режиме (без голоса)."""
        try:
            print("\n" + "="*60)
            print(f"🏥 МЕДИЦИНСКИЙ ЦЕНТР '{self.medical_center.upper()}' - ТЕКСТОВЫЙ РЕЖИМ")
            print(f"👩‍⚕️ Ваш помощник: {self.agent_name}")
            print("="*60)
            
            print(f"\n👋 Привет! Я {self.agent_name} из медицинского центра {self.medical_center}.")
            print("Помогу записаться на прием и отвечу на вопросы. Напишите ваш запрос:")
            
            conversation_count = 0
            
            while True:
                print(f"\n[{conversation_count + 1}] Ваш вопрос: ", end="")
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                # Проверяем команды выхода
                if any(word in user_input.lower() for word in 
                       ['пока', 'до свидания', 'выход', 'quit', 'стоп']):
                    print(f"\n👋 До свидания! Берегите здоровье!")
                    break
                
                # Генерируем и выводим ответ
                response = self.generate_response(user_input)
                print(f"\n🏥 {self.agent_name}: {response}")
                
                conversation_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
        except Exception as e:
            logger.error(f"Ошибка в текстовом режиме: {e}")
            print(f"💥 Ошибка: {e}")
    
    def show_statistics(self):
        """Показать статистику системы."""
        try:
            print("\n" + "="*50)
            print("📊 СТАТИСТИКА МЕДИЦИНСКОГО ЦЕНТРА")
            print("="*50)
            
            # Статистика базы данных
            if self.medical_db_service:
                db_stats = self.medical_db_service.get_statistics()
                print(f"👥 Всего врачей: {db_stats.get('total_doctors', 0)}")
                print(f"🏥 Всего услуг: {db_stats.get('total_services', 0)}")
                print(f"📅 Активных записей: {db_stats.get('active_appointments', 0)}")
                print(f"❌ Отмененных записей: {db_stats.get('cancelled_appointments', 0)}")
                
                specialties = db_stats.get('doctors_by_specialty', {})
                if specialties:
                    print("\n👨‍⚕️ Врачи по специальностям:")
                    for specialty, count in specialties.items():
                        print(f"  • {specialty}: {count} врач(ей)")
            
            # Статистика RAG
            if self.rag_service:
                rag_stats = self.rag_service.get_knowledge_stats()
                print(f"\n📚 База знаний: {rag_stats.get('files_in_directory', 0)} файлов")
                print(f"🔗 Векторное хранилище: {'✅ Активно' if rag_stats.get('vectorstore_initialized') else '❌ Неактивно'}")
            
            print("\n" + "="*50)
            
        except Exception as e:
            logger.error(f"Ошибка показа статистики: {e}")
            print(f"❌ Ошибка получения статистики: {e}")
    
    def close(self):
        """Закрытие всех сервисов."""
        try:
            print("\n🔄 Закрытие сервисов...")
            
            if self.stt_service:
                self.stt_service.close()
            
            if self.tts_service:
                self.tts_service.close()
            
            if self.rag_service:
                self.rag_service.close()
            
            if self.medical_db_service:
                self.medical_db_service.close()
            
            print("✅ Все сервисы закрыты")
            
        except Exception as e:
            logger.error(f"Ошибка закрытия сервисов: {e}")


def setup_parser():
    """Настройка аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Медицинский AI агент с поддержкой RAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py                          # Голосовой режим
  python main.py --text-mode              # Текстовый режим
  python main.py --stats                  # Показать статистику
  python main.py --name "Анна"            # Изменить имя агента
        """
    )
    
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
        help='Имя медицинского агента (по умолчанию: Марина)'
    )
    parser.add_argument(
        '--medical-center',
        default='Таблетка',
        help='Название медицинского центра (по умолчанию: Таблетка)'
    )
    parser.add_argument(
        '--text-mode',
        action='store_true',
        help='Запуск в текстовом режиме (без голоса)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Показать статистику и выйти'
    )
    parser.add_argument(
        '--log-level',
        default=LOG_LEVEL,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
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
    logger.info("🚀 Запуск медицинского AI агента с RAG")
    
    # Получаем API ключи
    yandex_api_key = args.yandex_api_key or YANDEX_API_KEY or os.getenv('YANDEX_API_KEY')
    openai_api_key = args.openai_api_key or OPENAI_API_KEY or os.getenv('OPENAI_API_KEY')
    
    if not yandex_api_key and not args.text_mode:
        print("❌ Ошибка: Не указан Yandex API ключ (нужен для голосового режима)")
        print("💡 Используйте --text-mode для работы без голоса")
        return 1
        
    if not openai_api_key:
        print("❌ Ошибка: Не указан OpenAI API ключ")
        print("💡 Установите переменную окружения OPENAI_API_KEY")
        return 1
    
    try:
        # Создаем ассистента
        assistant = MedicalVoiceAssistant(
            agent_name=args.name,
            medical_center=args.medical_center
        )
        
        # Инициализируем сервисы
        assistant.initialize_services(
            yandex_api_key=yandex_api_key or "",
            openai_api_key=openai_api_key
        )
        
        # Обрабатываем специальные режимы
        if args.stats:
            assistant.show_statistics()
            return 0
        
        # Запускаем нужный режим
        if args.text_mode:
            print("📝 Запуск в ТЕКСТОВОМ режиме")
            assistant.run_text_mode()
        else:
            print("🎤 Запуск в ГОЛОСОВОМ режиме")
            assistant.start_medical_conversation()
        
    except KeyboardInterrupt:
        print("\n👋 Программа завершена пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        print(f"💥 Критическая ошибка: {e}")
        return 1
    finally:
        # Закрываем сервисы
        try:
            if 'assistant' in locals():
                assistant.close()
        except:
            pass
            
    logger.info("👋 Работа медицинского агента завершена")
    return 0


if __name__ == "__main__":
    sys.exit(main())