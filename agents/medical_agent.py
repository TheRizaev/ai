"""
Медицинский агент с поддержкой RAG и специализированными функциями.
"""
import os
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent
from services.rag_service import RAGService
from services.medical_db_service import MedicalDBService
from services.appointment_service import AppointmentService
from data.prompts.medical_prompts import (
    MEDICAL_RECEPTIONIST_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    APPOINTMENT_BOOKING_PROMPT,
    MEDICAL_INFO_PROMPT,
    EMERGENCY_DETECTION_PROMPT
)

logger = logging.getLogger(__name__)

class MedicalAgent(BaseAgent):
    """Агент медицинского центра с поддержкой RAG и специализированными функциями."""
    
    def __init__(self, name: str = "Марина", medical_center_name: str = "Таблетка"):
        """
        Инициализация медицинского агента.
        
        Args:
            name: Имя агента
            medical_center_name: Название медицинского центра
        """
        super().__init__(name)
        
        self.medical_center_name = medical_center_name
        self.conversation_memory = ConversationBufferWindowMemory(
            k=10,  # Храним последние 10 сообщений
            return_messages=True
        )
        
        # Специализированные сервисы
        self.rag_service = None
        self.medical_db_service = None
        self.appointment_service = None
        
        # Контекст текущего разговора
        self.current_intent = None
        self.appointment_context = {}
        self.patient_info = {}
        
        logger.info(f"Создан медицинский агент '{name}' для центра '{medical_center_name}'")
    
    def register_services(self, rag_service: RAGService, 
                         medical_db_service: MedicalDBService,
                         appointment_service: AppointmentService):
        """Регистрация специализированных сервисов."""
        self.rag_service = rag_service
        self.medical_db_service = medical_db_service
        self.appointment_service = appointment_service
        
        logger.info("Медицинские сервисы зарегистрированы")
    
    def detect_intent(self, user_message: str) -> str:
        """
        Определение намерения пользователя.
        
        Returns:
            Тип намерения: greeting, appointment, services_info, price_inquiry,
            medical_question, emergency, complaint, schedule_inquiry, goodbye
        """
        if not self.llm_service:
            return "unknown"
        
        try:
            # Создаем промпт для классификации намерения
            intent_prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT)
            
            # Используем LLM для классификации
            chain = intent_prompt | self.llm_service.llm | StrOutputParser()
            
            result = chain.invoke({
                "user_message": user_message,
                "medical_center": self.medical_center_name
            })
            
            # Извлекаем намерение из ответа
            intent = result.strip().lower()
            
            # Валидируем намерение
            valid_intents = [
                "greeting", "appointment", "services_info", "price_inquiry",
                "medical_question", "emergency", "complaint", "schedule_inquiry", 
                "goodbye", "unknown"
            ]
            
            if intent not in valid_intents:
                intent = "unknown"
            
            self.current_intent = intent
            logger.info(f"Определено намерение: {intent}")
            
            return intent
            
        except Exception as e:
            logger.error(f"Ошибка определения намерения: {e}")
            return "unknown"
    
    def check_emergency(self, user_message: str) -> bool:
        """
        Проверка на экстренную ситуацию.
        
        Returns:
            True если обнаружена экстренная ситуация
        """
        emergency_keywords = [
            "сильная боль", "не могу дышать", "кровотечение", "потерял сознание",
            "сердечный приступ", "инсульт", "температура 40", "судороги",
            "аллергическая реакция", "отравление", "травма", "перелом"
        ]
        
        user_lower = user_message.lower()
        
        # Простая проверка по ключевым словам
        for keyword in emergency_keywords:
            if keyword in user_lower:
                logger.warning(f"Обнаружена потенциальная экстренная ситуация: {keyword}")
                return True
        
        # Дополнительно используем LLM для более точной оценки
        if self.llm_service:
            try:
                emergency_prompt = ChatPromptTemplate.from_template(EMERGENCY_DETECTION_PROMPT)
                chain = emergency_prompt | self.llm_service.llm | StrOutputParser()
                
                result = chain.invoke({"user_message": user_message})
                
                if "ЭКСТРЕННО" in result.upper():
                    logger.warning("LLM обнаружил экстренную ситуацию")
                    return True
                    
            except Exception as e:
                logger.error(f"Ошибка проверки экстренной ситуации: {e}")
        
        return False
    
    def handle_appointment_booking(self, user_message: str) -> str:
        """Обработка записи на прием."""
        if not self.appointment_service:
            return "Извините, сервис записи временно недоступен."
        
        try:
            # Извлекаем информацию о записи из сообщения
            appointment_info = self._extract_appointment_info(user_message)
            
            # Дополняем контекст записи
            self.appointment_context.update(appointment_info)
            
            # Проверяем, какая информация еще нужна
            missing_info = self._get_missing_appointment_info()
            
            if missing_info:
                return self._ask_for_missing_info(missing_info)
            
            # Пытаемся забронировать прием
            booking_result = self.appointment_service.book_appointment(
                doctor_specialty=self.appointment_context.get('specialty'),
                doctor_name=self.appointment_context.get('doctor_name'),
                date=self.appointment_context.get('date'),
                time=self.appointment_context.get('time'),
                patient_name=self.appointment_context.get('patient_name'),
                patient_phone=self.appointment_context.get('patient_phone'),
                patient_complaint=self.appointment_context.get('complaint')
            )
            
            if booking_result['success']:
                # Очищаем контекст после успешной записи
                self.appointment_context = {}
                return f"Отлично! Я записала вас на прием. {booking_result['message']}"
            else:
                return f"К сожалению, не удалось записать на прием: {booking_result['message']}"
                
        except Exception as e:
            logger.error(f"Ошибка записи на прием: {e}")
            return "Извините, произошла ошибка при записи. Попробуйте еще раз."
    
    def _extract_appointment_info(self, user_message: str) -> Dict[str, Any]:
        """Извлечение информации о записи из сообщения пользователя."""
        if not self.llm_service:
            return {}
        
        try:
            extraction_prompt = ChatPromptTemplate.from_template(APPOINTMENT_BOOKING_PROMPT)
            chain = extraction_prompt | self.llm_service.llm | StrOutputParser()
            
            result = chain.invoke({
                "user_message": user_message,
                "current_context": str(self.appointment_context)
            })
            
            # Парсим результат (предполагается JSON-формат)
            import json
            try:
                info = json.loads(result)
                return info
            except json.JSONDecodeError:
                # Если не JSON, пытаемся извлечь информацию по-другому
                return self._parse_appointment_info_fallback(result)
                
        except Exception as e:
            logger.error(f"Ошибка извлечения информации о записи: {e}")
            return {}
    
    def _parse_appointment_info_fallback(self, text: str) -> Dict[str, Any]:
        """Запасной метод парсинга информации о записи."""
        info = {}
        
        # Простой поиск по ключевым словам
        text_lower = text.lower()
        
        # Специальности
        specialties_map = {
            'терапевт': 'therapy',
            'кардиолог': 'cardiology',
            'невролог': 'neurology',
            'гинеколог': 'gynecology',
            'уролог': 'urology',
            'педиатр': 'pediatrics'
        }
        
        for russian, english in specialties_map.items():
            if russian in text_lower:
                info['specialty'] = english
                break
        
        # Дата (упрощенный поиск)
        import re
        date_patterns = [
            r'(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)',
            r'(\d{1,2})\.(\d{1,2})\.(\d{4})',
            r'завтра', r'послезавтра', r'сегодня'
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text_lower):
                info['date_raw'] = re.search(pattern, text_lower).group()
                break
        
        return info
    
    def _get_missing_appointment_info(self) -> List[str]:
        """Определение недостающей информации для записи."""
        required_fields = ['specialty', 'patient_name', 'patient_phone']
        missing = []
        
        for field in required_fields:
            if field not in self.appointment_context or not self.appointment_context[field]:
                missing.append(field)
        
        return missing
    
    def _ask_for_missing_info(self, missing_info: List[str]) -> str:
        """Запрос недостающей информации."""
        questions = {
            'specialty': "К какому специалисту вы хотите записаться?",
            'patient_name': "Скажите, пожалуйста, ваше имя.",
            'patient_phone': "Укажите ваш контактный телефон для записи.",
            'date': "На какую дату вам удобно прийти?",
            'time': "В какое время вам будет удобно?"
        }
        
        if len(missing_info) == 1:
            return questions.get(missing_info[0], "Мне нужна дополнительная информация.")
        else:
            return f"Мне нужна еще информация: {', '.join([questions.get(field, field) for field in missing_info[:2]])}."
    
    def handle_medical_question(self, user_message: str) -> str:
        """Обработка медицинских вопросов с использованием RAG."""
        if not self.rag_service:
            return "К сожалению, я не могу дать медицинские консультации. Рекомендую обратиться к врачу."
        
        try:
            # Поиск релевантной информации в медицинской базе знаний
            relevant_docs = self.rag_service.search_medical_knowledge(
                query=user_message,
                top_k=3
            )
            
            if not relevant_docs:
                return "К сожалению, я не нашла информации по вашему вопросу. Рекомендую обратиться к специалисту."
            
            # Формируем ответ на основе найденной информации
            medical_prompt = ChatPromptTemplate.from_template(MEDICAL_INFO_PROMPT)
            
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            chain = medical_prompt | self.llm_service.llm | StrOutputParser()
            
            response = chain.invoke({
                "user_question": user_message,
                "medical_context": context,
                "agent_name": self.name,
                "medical_center": self.medical_center_name
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка обработки медицинского вопроса: {e}")
            return "Извините, произошла ошибка. Лучше обратиться к врачу за консультацией."
    
    def handle_services_inquiry(self, user_message: str) -> str:
        """Обработка запросов об услугах."""
        if not self.medical_db_service:
            return "Информация об услугах временно недоступна."
        
        try:
            # Получаем информацию об услугах
            services = self.medical_db_service.get_services_info(user_message)
            
            if not services:
                return "К сожалению, не нашла информации об этой услуге. Уточните, пожалуйста, что вас интересует."
            
            # Формируем ответ
            response = "Вот информация об услугах нашего центра:\n\n"
            for service in services:
                response += f"• {service['name']} - {service['price']} руб.\n"
                if service.get('description'):
                    response += f"  {service['description']}\n"
            
            response += "\nХотите записаться на какую-то из этих услуг?"
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка получения информации об услугах: {e}")
            return "Извините, не могу получить информацию об услугах."
    
    def generate_response(self, user_message: str) -> str:
        """
        Основной метод генерации ответа.
        
        Args:
            user_message: Сообщение пользователя
            
        Returns:
            Ответ агента
        """
        try:
            # 1. Проверяем на экстренную ситуацию
            if self.check_emergency(user_message):
                return ("⚠️ Это похоже на экстренную ситуацию! Немедленно обратитесь в службу "
                       "скорой помощи по номеру 103 или в ближайшее отделение неотложной помощи!")
            
            # 2. Определяем намерение
            intent = self.detect_intent(user_message)
            
            # 3. Обрабатываем в зависимости от намерения
            if intent == "greeting":
                return self._handle_greeting()
            
            elif intent == "appointment":
                return self.handle_appointment_booking(user_message)
            
            elif intent == "services_info" or intent == "price_inquiry":
                return self.handle_services_inquiry(user_message)
            
            elif intent == "medical_question":
                return self.handle_medical_question(user_message)
            
            elif intent == "schedule_inquiry":
                return self._handle_schedule_inquiry()
            
            elif intent == "goodbye":
                return self._handle_goodbye()
            
            elif intent == "complaint":
                return self._handle_complaint(user_message)
            
            else:
                # Общий ответ с использованием медицинского контекста
                return self._handle_general_inquiry(user_message)
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return "Извините, произошла ошибка. Не могли бы вы повторить свой вопрос?"
    
    def _handle_greeting(self) -> str:
        """Обработка приветствия."""
        greetings = [
            f"Добро пожаловать в медицинский центр '{self.medical_center_name}'! Меня зовут {self.name}, я помогу вам с записью на прием и отвечу на вопросы об наших услугах.",
            f"Здравствуйте! Это медицинский центр '{self.medical_center_name}', {self.name} на связи. Чем могу помочь?",
            f"Добрый день! {self.name} из медицинского центра '{self.medical_center_name}'. Как дела с вашим здоровьем? Чем могу быть полезна?"
        ]
        
        import random
        return random.choice(greetings)
    
    def _handle_schedule_inquiry(self) -> str:
        """Обработка запросов о расписании."""
        return (f"Медицинский центр '{self.medical_center_name}' работает:\n"
                "• Понедельник-пятница: 8:00-20:00\n"
                "• Суббота: 9:00-18:00\n"
                "• Воскресенье: 9:00-15:00\n\n"
                "Процедурный кабинет: ежедневно 8:00-12:00\n"
                "Хотите записаться на прием?")
    
    def _handle_goodbye(self) -> str:
        """Обработка прощания."""
        goodbyes = [
            "До свидания! Берегите здоровье и обращайтесь, если понадобится помощь!",
            "Всего доброго! Будьте здоровы!",
            f"До встречи! Спасибо, что выбрали медицинский центр '{self.medical_center_name}'!"
        ]
        
        import random
        return random.choice(goodbyes)
    
    def _handle_complaint(self, user_message: str) -> str:
        """Обработка жалоб."""
        return ("Я понимаю ваше беспокойство. Жалобы и предложения очень важны для нас. "
                "Я передам вашу информацию администрации. Также вы можете обратиться "
                "напрямую к главврачу или оставить отзыв на нашем сайте. "
                "Чем еще могу помочь?")
    
    def _handle_general_inquiry(self, user_message: str) -> str:
        """Обработка общих запросов."""
        if not self.llm_service:
            return "Извините, не могу обработать ваш запрос. Уточните, пожалуйста."
        
        try:
            # Получаем историю разговора
            memory_context = self.conversation_memory.load_memory_variables({})
            history = memory_context.get('history', [])
            
            # Создаем промпт с медицинским контекстом
            system_prompt = MEDICAL_RECEPTIONIST_SYSTEM_PROMPT.format(
                agent_name=self.name,
                medical_center=self.medical_center_name
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{user_message}")
            ])
            
            chain = prompt | self.llm_service.llm | StrOutputParser()
            
            response = chain.invoke({"user_message": user_message})
            
            # Сохраняем в память
            self.conversation_memory.save_context(
                {"input": user_message},
                {"output": response}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка обработки общего запроса: {e}")
            return "Извините, не совсем поняла ваш вопрос. Могли бы вы уточнить?"