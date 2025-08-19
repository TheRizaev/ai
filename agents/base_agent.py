"""
Базовый класс для всех агентов.
"""
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Базовый класс для всех агентов."""
    
    def __init__(self, name: str):
        """
        Инициализация базового агента.
        
        Args:
            name: Имя агента
        """
        self.name = name
        self.stt_service = None
        self.tts_service = None
        self.llm_service = None
        
        logger.info(f"Создан базовый агент: {name}")
    
    def connect_services(self, stt_service=None, tts_service=None, llm_service=None):
        """
        Подключение базовых сервисов.
        
        Args:
            stt_service: Сервис распознавания речи
            tts_service: Сервис синтеза речи
            llm_service: Сервис языковой модели
        """
        if stt_service:
            self.stt_service = stt_service
            
        if tts_service:
            self.tts_service = tts_service
            
        if llm_service:
            self.llm_service = llm_service
        
        logger.info(f"Сервисы подключены к агенту {self.name}")
    
    @abstractmethod
    def generate_response(self, user_message: str) -> str:
        """
        Абстрактный метод для генерации ответа.
        
        Args:
            user_message: Сообщение пользователя
            
        Returns:
            Ответ агента
        """
        pass
    
    def listen(self) -> Optional[str]:
        """
        Слушать пользователя (если подключен STT сервис).
        
        Returns:
            Распознанный текст или None
        """
        if not self.stt_service:
            logger.warning("STT сервис не подключен")
            return None
        
        try:
            return self.stt_service.recognize_stream()
        except Exception as e:
            logger.error(f"Ошибка распознавания речи: {e}")
            return None
    
    def speak(self, message: str):
        """
        Произнести сообщение (если подключен TTS сервис).
        
        Args:
            message: Сообщение для произношения
        """
        if not self.tts_service:
            logger.warning("TTS сервис не подключен")
            print(f"{self.name}: {message}")
            return
        
        try:
            audio = self.tts_service.synthesize(message)
            if audio:
                from utils.audio_utils import AudioPlayer
                AudioPlayer.play_audio_segment(audio)
        except Exception as e:
            logger.error(f"Ошибка синтеза речи: {e}")
            print(f"{self.name}: {message}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Получение информации об агенте.
        
        Returns:
            Информация об агенте
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "stt_connected": self.stt_service is not None,
            "tts_connected": self.tts_service is not None,
            "llm_connected": self.llm_service is not None
        }