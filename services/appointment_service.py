"""
Сервис для управления записями на прием.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

class AppointmentService:
    """Сервис для записи на прием и управления расписанием."""
    
    def __init__(self, medical_db_service):
        """
        Инициализация сервиса записи.
        
        Args:
            medical_db_service: Сервис медицинской базы данных
        """
        self.db_service = medical_db_service
        
        # Маппинг специальностей
        self.specialty_mapping = {
            "терапевт": "therapy",
            "терапия": "therapy",
            "терапевта": "therapy",
            "кардиолог": "cardiology",
            "кардиология": "cardiology",
            "кардиолога": "cardiology",
            "невролог": "neurology",
            "неврология": "neurology",
            "невролога": "neurology",
            "гинеколог": "gynecology",
            "гинекология": "gynecology",
            "гинеколога": "gynecology",
            "уролог": "urology",
            "урология": "urology",
            "уролога": "urology",
            "педиатр": "pediatrics",
            "педиатрия": "pediatrics",
            "педиатра": "pediatrics"
        }
        
        # Маппинг дней недели
        self.weekday_mapping = {
            "понедельник": "monday",
            "вторник": "tuesday", 
            "среда": "wednesday",
            "четверг": "thursday",
            "пятница": "friday",
            "суббота": "saturday",
            "воскресенье": "sunday",
            "пн": "monday",
            "вт": "tuesday",
            "ср": "wednesday", 
            "чт": "thursday",
            "пт": "friday",
            "сб": "saturday",
            "вс": "sunday"
        }
        
        # Маппинг месяцев
        self.month_mapping = {
            "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
            "мая": 5, "июня": 6, "июля": 7, "августа": 8,
            "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
        }
        
        logger.info("Сервис записи на прием инициализирован")
    
    def parse_date_from_text(self, text: str) -> Optional[str]:
        """
        Извлечение даты из текста.
        
        Args:
            text: Текст с датой
            
        Returns:
            Дата в формате YYYY-MM-DD или None
        """
        try:
            text_lower = text.lower()
            
            # Проверяем относительные даты
            if "сегодня" in text_lower:
                return datetime.now().strftime("%Y-%m-%d")
            elif "завтра" in text_lower:
                return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            elif "послезавтра" in text_lower:
                return (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
            
            # Проверяем дни недели
            for russian_day, english_day in self.weekday_mapping.items():
                if russian_day in text_lower:
                    return self._get_next_weekday_date(english_day)
            
            # Проверяем числовые форматы даты
            date_patterns = [
                r"(\d{1,2})\.(\d{1,2})\.(\d{4})",  # ДД.ММ.ГГГГ
                r"(\d{1,2})\.(\d{1,2})",           # ДД.ММ
                r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return self._parse_date_match(match, pattern)
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка парсинга даты: {e}")
            return None
    
    def _get_next_weekday_date(self, weekday: str) -> str:
        """Получение ближайшей даты для указанного дня недели."""
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        
        if weekday not in weekdays:
            return datetime.now().strftime("%Y-%m-%d")
        
        target_weekday = weekdays.index(weekday)
        current_weekday = datetime.now().weekday()
        
        days_ahead = target_weekday - current_weekday
        if days_ahead <= 0:
            days_ahead += 7
        
        target_date = datetime.now() + timedelta(days=days_ahead)
        return target_date.strftime("%Y-%m-%d")
    
    def _parse_date_match(self, match, pattern: str) -> str:
        """Парсинг даты из regex match."""
        try:
            if "января|февраля" in pattern:
                # Формат "15 января"
                day = int(match.group(1))
                month_name = match.group(2)
                
                month = self.month_mapping.get(month_name, 1)
                year = datetime.now().year
                
                date_obj = datetime(year, month, day)
                
                # Если дата уже прошла в этом году, берем следующий год
                if date_obj < datetime.now():
                    date_obj = datetime(year + 1, month, day)
                
                return date_obj.strftime("%Y-%m-%d")
            
            elif "(\d{1,2})\.(\d{1,2})\.(\d{4})" in pattern:
                # Формат "15.03.2024"
                day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                date_obj = datetime(year, month, day)
                return date_obj.strftime("%Y-%m-%d")
            
            elif "(\d{1,2})\.(\d{1,2})" in pattern:
                # Формат "15.03"
                day, month = int(match.group(1)), int(match.group(2))
                year = datetime.now().year
                
                date_obj = datetime(year, month, day)
                
                # Если дата уже прошла в этом году, берем следующий год
                if date_obj < datetime.now():
                    date_obj = datetime(year + 1, month, day)
                
                return date_obj.strftime("%Y-%m-%d")
            
            return datetime.now().strftime("%Y-%m-%d")
            
        except Exception as e:
            logger.error(f"Ошибка парсинга даты из match: {e}")
            return datetime.now().strftime("%Y-%m-%d")
    
    def parse_time_from_text(self, text: str) -> Optional[str]:
        """
        Извлечение времени из текста.
        
        Args:
            text: Текст со временем
            
        Returns:
            Время в формате HH:MM или None
        """
        try:
            time_patterns = [
                r"(\d{1,2}):(\d{2})",           # 14:30
                r"(\d{1,2})\.(\d{2})",          # 14.30
                r"в\s*(\d{1,2})",               # в 14
                r"(\d{1,2})\s*утра",            # 9 утра
                r"(\d{1,2})\s*дня",             # 2 дня
                r"(\d{1,2})\s*вечера"           # 6 вечера
            ]
            
            text_lower = text.lower()
            
            for pattern in time_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return self._parse_time_match(match, pattern)
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка парсинга времени: {e}")
            return None
    
    def _parse_time_match(self, match, pattern: str) -> str:
        """Парсинг времени из regex match."""
        try:
            if ":(\d{2})" in pattern or "\.(\d{2})" in pattern:
                # Формат 14:30 или 14.30
                hour = int(match.group(1))
                minute = int(match.group(2))
                return f"{hour:02d}:{minute:02d}"
            
            elif "в\s*(\d{1,2})" in pattern:
                # Формат "в 14"
                hour = int(match.group(1))
                return f"{hour:02d}:00"
            
            elif "утра" in pattern:
                # Формат "9 утра"
                hour = int(match.group(1))
                if hour == 12:
                    hour = 0
                return f"{hour:02d}:00"
            
            elif "дня" in pattern:
                # Формат "2 дня"
                hour = int(match.group(1))
                if hour < 12:
                    hour += 12
                return f"{hour:02d}:00"
            
            elif "вечера" in pattern:
                # Формат "6 вечера"
                hour = int(match.group(1))
                if hour < 12:
                    hour += 12
                return f"{hour:02d}:00"
            
            return "09:00"
            
        except Exception as e:
            logger.error(f"Ошибка парсинга времени: {e}")
            return "09:00"
    
    def normalize_specialty(self, specialty_text: str) -> Optional[str]:
        """Нормализация названия специальности."""
        text_lower = specialty_text.lower()
        
        for russian_specialty, english_specialty in self.specialty_mapping.items():
            if russian_specialty in text_lower:
                return english_specialty
        
        return None
    
    def book_appointment(self, doctor_specialty: str = None, doctor_name: str = None,
                        date: str = None, time: str = None, patient_name: str = None,
                        patient_phone: str = None, patient_complaint: str = None) -> Dict[str, Any]:
        """
        Запись на прием к врачу.
        
        Returns:
            Результат записи
        """
        try:
            # Нормализуем специальность
            if doctor_specialty:
                specialty = self.normalize_specialty(doctor_specialty)
                if not specialty:
                    return {
                        "success": False,
                        "message": "Неизвестная специальность. Доступные: терапевт, кардиолог, невролог, гинеколог, уролог, педиатр."
                    }
            else:
                return {
                    "success": False,
                    "message": "Не указана специальность врача"
                }
            
            # Находим врачей нужной специальности
            doctors = self.db_service.get_doctors_by_specialty(specialty)
            
            if not doctors:
                return {
                    "success": False,
                    "message": f"Врачи специальности '{doctor_specialty}' не найдены"
                }
            
            # Выбираем врача
            selected_doctor = None
            
            if doctor_name:
                # Ищем конкретного врача
                for doctor in doctors:
                    if doctor_name.lower() in doctor.get("name", "").lower():
                        selected_doctor = doctor
                        break
                
                if not selected_doctor:
                    return {
                        "success": False,
                        "message": f"Врач '{doctor_name}' не найден среди {doctor_specialty}ов"
                    }
            else:
                # Берем первого доступного врача
                selected_doctor = doctors[0]
            
            doctor_id = selected_doctor.get("id")
            
            # Проверяем и корректируем дату
            if not date:
                # Ищем ближайшую доступную дату
                date = self._find_next_available_date(doctor_id)
            
            # Проверяем и корректируем время
            if not time:
                # Ищем ближайшее доступное время
                available_times = self.db_service.get_available_times(doctor_id, date)
                if available_times:
                    time = available_times[0]
                else:
                    return {
                        "success": False,
                        "message": f"На дату {date} нет свободного времени у врача {selected_doctor.get('name')}"
                    }
            
            # Проверяем доступность времени
            is_available = self.db_service.check_appointment_availability(doctor_id, date, time)
            
            if not is_available:
                # Предлагаем альтернативное время
                available_times = self.db_service.get_available_times(doctor_id, date)
                
                if available_times:
                    suggested_times = ", ".join(available_times[:3])
                    return {
                        "success": False,
                        "message": f"Время {time} занято. Доступное время: {suggested_times}"
                    }
                else:
                    return {
                        "success": False,
                        "message": f"На дату {date} нет свободного времени. Выберите другую дату."
                    }
            
            # Проверяем обязательные поля
            if not patient_name:
                return {
                    "success": False,
                    "message": "Укажите ваше имя для записи"
                }
            
            if not patient_phone:
                return {
                    "success": False,
                    "message": "Укажите контактный телефон"
                }
            
            # Ищем или создаем пациента
            patient = self.db_service.find_patient_by_phone(patient_phone)
            
            if not patient:
                patient_id = self.db_service.add_patient({
                    "name": patient_name,
                    "phone": patient_phone
                })
            else:
                patient_id = patient.get("id")
            
            # Создаем запись
            appointment_data = {
                "doctor_id": doctor_id,
                "patient_name": patient_name,
                "patient_phone": patient_phone,
                "date": date,
                "time": time,
                "complaint": patient_complaint or "",
                "service_id": f"{specialty}_consult"
            }
            
            result = self.db_service.create_appointment(appointment_data)
            
            if result.get("success"):
                doctor_name = selected_doctor.get("name")
                return {
                    "success": True,
                    "appointment_id": result.get("appointment_id"),
                    "message": f"Отлично! Записала вас к врачу {doctor_name} на {date} в {time}. Кабинет {selected_doctor.get('room', 'уточните на ресепшн')}."
                }
            else:
                return result
            
        except Exception as e:
            logger.error(f"Ошибка записи на прием: {e}")
            return {
                "success": False,
                "message": "Произошла ошибка при записи. Попробуйте еще раз."
            }
    
    def _find_next_available_date(self, doctor_id: str, max_days: int = 14) -> str:
        """Поиск ближайшей доступной даты для врача."""
        current_date = datetime.now()
        
        for i in range(max_days):
            check_date = current_date + timedelta(days=i)
            date_str = check_date.strftime("%Y-%m-%d")
            
            # Получаем доступное время на эту дату
            available_times = self.db_service.get_available_times(doctor_id, date_str)
            
            if available_times:
                return date_str
        
        # Если не найдено доступных дат, возвращаем завтра
        return (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    def get_doctor_availability(self, doctor_specialty: str, date: str = None) -> Dict[str, Any]:
        """Получение информации о доступности врачей."""
        try:
            specialty = self.normalize_specialty(doctor_specialty)
            
            if not specialty:
                return {
                    "success": False,
                    "message": "Неизвестная специальность"
                }
            
            doctors = self.db_service.get_doctors_by_specialty(specialty)
            
            if not doctors:
                return {
                    "success": False,
                    "message": f"Врачи специальности '{doctor_specialty}' не найдены"
                }
            
            if not date:
                date = datetime.now().strftime("%Y-%m-%d")
            
            availability_info = []
            
            for doctor in doctors:
                doctor_id = doctor.get("id")
                available_times = self.db_service.get_available_times(doctor_id, date)
                
                availability_info.append({
                    "doctor_name": doctor.get("name"),
                    "room": doctor.get("room"),
                    "available_times": available_times,
                    "total_slots": len(available_times)
                })
            
            return {
                "success": True,
                "date": date,
                "specialty": doctor_specialty,
                "doctors": availability_info
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения доступности: {e}")
            return {
                "success": False,
                "message": "Ошибка при получении информации о доступности"
            }
    
    def suggest_alternative_appointments(self, doctor_specialty: str, 
                                       preferred_date: str = None,
                                       preferred_time: str = None) -> Dict[str, Any]:
        """Предложение альтернативных вариантов записи."""
        try:
            specialty = self.normalize_specialty(doctor_specialty)
            
            if not specialty:
                return {"success": False, "message": "Неизвестная специальность"}
            
            doctors = self.db_service.get_doctors_by_specialty(specialty)
            
            if not doctors:
                return {"success": False, "message": "Врачи не найдены"}
            
            suggestions = []
            current_date = datetime.now()
            
            # Ищем варианты на ближайшие 7 дней
            for i in range(7):
                check_date = current_date + timedelta(days=i)
                date_str = check_date.strftime("%Y-%m-%d")
                
                for doctor in doctors:
                    doctor_id = doctor.get("id")
                    available_times = self.db_service.get_available_times(doctor_id, date_str)
                    
                    if available_times:
                        suggestions.append({
                            "date": date_str,
                            "doctor_name": doctor.get("name"),
                            "doctor_id": doctor_id,
                            "available_times": available_times[:3],
                            "room": doctor.get("room")
                        })
                
                # Ограничиваем количество предложений
                if len(suggestions) >= 5:
                    break
            
            return {
                "success": True,
                "suggestions": suggestions[:5],
                "total_found": len(suggestions)
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения альтернатив: {e}")
            return {
                "success": False,
                "message": "Ошибка при поиске альтернативных вариантов"
            }