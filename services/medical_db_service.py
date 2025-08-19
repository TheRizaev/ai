"""
Сервис для работы с медицинской базой данных центра.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class MedicalDBService:
    """Сервис для работы с базой данных медицинского центра."""
    
    def __init__(self, db_path: str = None):
        """
        Инициализация сервиса.
        
        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = db_path or self._get_default_db_path()
        self.data = {
            "services": [],
            "doctors": [],
            "schedule": {},
            "appointments": [],
            "patients": []
        }
        
        self._load_data()
        self._ensure_sample_data()
    
    def _get_default_db_path(self) -> str:
        """Получить путь к базе данных по умолчанию."""
        current_dir = Path(__file__).resolve().parent.parent
        data_dir = current_dir / "data" / "database"
        data_dir.mkdir(parents=True, exist_ok=True)
        return str(data_dir / "medical_center.json")
    
    def _load_data(self):
        """Загрузка данных из файла."""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                logger.info(f"Данные загружены из {self.db_path}")
            else:
                logger.info("Файл базы данных не найден, создается новый")
                self._save_data()
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            self.data = {
                "services": [],
                "doctors": [],
                "schedule": {},
                "appointments": [],
                "patients": []
            }
    
    def _save_data(self):
        """Сохранение данных в файл."""
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            logger.debug("Данные сохранены в базу")
        except Exception as e:
            logger.error(f"Ошибка сохранения данных: {e}")
    
    def _ensure_sample_data(self):
        """Создание примеров данных, если база пуста."""
        if not self.data.get("services"):
            self._create_sample_services()
        
        if not self.data.get("doctors"):
            self._create_sample_doctors()
        
        if not self.data.get("schedule"):
            self._create_sample_schedule()
        
        self._save_data()
    
    def _create_sample_services(self):
        """Создание примеров услуг."""
        sample_services = [
            {
                "id": "therapy_consult",
                "name": "Консультация терапевта",
                "category": "therapy",
                "price": 2500,
                "duration": 30,
                "description": "Первичная консультация врача-терапевта",
                "preparation": "Особой подготовки не требуется"
            },
            {
                "id": "therapy_repeat",
                "name": "Повторная консультация терапевта",
                "category": "therapy",
                "price": 2000,
                "duration": 20,
                "description": "Повторный прием врача-терапевта"
            },
            {
                "id": "cardio_consult",
                "name": "Консультация кардиолога",
                "category": "cardiology",
                "price": 3000,
                "duration": 40,
                "description": "Консультация врача-кардиолога"
            },
            {
                "id": "neuro_consult",
                "name": "Консультация невролога",
                "category": "neurology",
                "price": 2800,
                "duration": 35,
                "description": "Консультация врача-невролога"
            },
            {
                "id": "gyneco_consult",
                "name": "Консультация гинеколога",
                "category": "gynecology",
                "price": 2500,
                "duration": 30,
                "description": "Консультация врача-гинеколога"
            },
            {
                "id": "urology_consult",
                "name": "Консультация уролога",
                "category": "urology",
                "price": 2700,
                "duration": 30,
                "description": "Консультация врача-уролога"
            },
            {
                "id": "pediatric_consult",
                "name": "Консультация педиатра",
                "category": "pediatrics",
                "price": 2200,
                "duration": 30,
                "description": "Консультация врача-педиатра"
            },
            {
                "id": "ecg",
                "name": "ЭКГ с расшифровкой",
                "category": "diagnostics",
                "price": 800,
                "duration": 15,
                "description": "Электрокардиография с расшифровкой"
            },
            {
                "id": "echo_heart",
                "name": "УЗИ сердца (ЭхоКГ)",
                "category": "diagnostics",
                "price": 2800,
                "duration": 30,
                "description": "Ультразвуковое исследование сердца"
            },
            {
                "id": "blood_general",
                "name": "Общий анализ крови",
                "category": "laboratory",
                "price": 500,
                "duration": 10,
                "description": "Общий клинический анализ крови",
                "preparation": "Сдается натощак утром"
            },
            {
                "id": "blood_biochem",
                "name": "Биохимический анализ крови",
                "category": "laboratory",
                "price": 1200,
                "duration": 10,
                "description": "Биохимическое исследование крови",
                "preparation": "12-часовое голодание перед сдачей"
            },
            {
                "id": "urine_general",
                "name": "Общий анализ мочи",
                "category": "laboratory",
                "price": 400,
                "duration": 5,
                "description": "Общий клинический анализ мочи",
                "preparation": "Утренняя порция мочи, средний поток"
            }
        ]
        
        self.data["services"] = sample_services
        logger.info("Созданы примеры услуг")
    
    def _create_sample_doctors(self):
        """Создание примеров врачей."""
        sample_doctors = [
            {
                "id": "ivanova_ap",
                "name": "Иванова Анна Петровна",
                "specialty": "therapy",
                "position": "Врач-терапевт",
                "experience": 15,
                "education": "ТГМИ, 2009",
                "room": "101"
            },
            {
                "id": "petrov_sm",
                "name": "Петров Сергей Михайлович",
                "specialty": "therapy",
                "position": "Врач-терапевт высшей категории",
                "experience": 20,
                "education": "ТГМИ, 2004",
                "room": "102"
            },
            {
                "id": "sidorova_ev",
                "name": "Сидорова Елена Владимировна",
                "specialty": "cardiology",
                "position": "Врач-кардиолог",
                "experience": 12,
                "education": "ТГМИ, 2012",
                "room": "201"
            },
            {
                "id": "mikhailov_ik",
                "name": "Михайлов Игорь Константинович",
                "specialty": "cardiology",
                "position": "Врач-кардиолог высшей категории",
                "experience": 25,
                "education": "ТГМИ, 1999",
                "room": "202"
            },
            {
                "id": "kozlova_ma",
                "name": "Козлова Мария Александровна",
                "specialty": "neurology",
                "position": "Врач-невролог",
                "experience": 10,
                "education": "ТГМИ, 2014",
                "room": "301"
            },
            {
                "id": "fedorov_ds",
                "name": "Федоров Дмитрий Сергеевич",
                "specialty": "neurology",
                "position": "Врач-невролог",
                "experience": 8,
                "education": "ТГМИ, 2016",
                "room": "302"
            },
            {
                "id": "romanova_li",
                "name": "Романова Людмила Ивановна",
                "specialty": "gynecology",
                "position": "Врач-гинеколог",
                "experience": 18,
                "education": "ТГМИ, 2006",
                "room": "401"
            },
            {
                "id": "nikolaeva_op",
                "name": "Николаева Ольга Павловна",
                "specialty": "gynecology",
                "position": "Врач-гинеколог высшей категории",
                "experience": 22,
                "education": "ТГМИ, 2002",
                "room": "402"
            }
        ]
        
        self.data["doctors"] = sample_doctors
        logger.info("Созданы примеры врачей")
    
    def _create_sample_schedule(self):
        """Создание примера расписания."""
        schedule = {
            "ivanova_ap": {
                "monday": {"start": "09:00", "end": "15:00"},
                "wednesday": {"start": "09:00", "end": "15:00"},
                "friday": {"start": "09:00", "end": "15:00"}
            },
            "petrov_sm": {
                "tuesday": {"start": "10:00", "end": "16:00"},
                "thursday": {"start": "10:00", "end": "16:00"},
                "saturday": {"start": "09:00", "end": "13:00"}
            },
            "sidorova_ev": {
                "monday": {"start": "14:00", "end": "19:00"},
                "wednesday": {"start": "14:00", "end": "19:00"}
            },
            "mikhailov_ik": {
                "tuesday": {"start": "09:00", "end": "14:00"},
                "thursday": {"start": "09:00", "end": "14:00"},
                "friday": {"start": "09:00", "end": "14:00"}
            },
            "kozlova_ma": {
                "monday": {"start": "10:00", "end": "16:00"},
                "tuesday": {"start": "10:00", "end": "16:00"},
                "thursday": {"start": "10:00", "end": "16:00"}
            },
            "fedorov_ds": {
                "wednesday": {"start": "14:00", "end": "20:00"},
                "friday": {"start": "14:00", "end": "20:00"}
            },
            "romanova_li": {
                "tuesday": {"start": "09:00", "end": "15:00"},
                "thursday": {"start": "09:00", "end": "15:00"},
                "saturday": {"start": "09:00", "end": "15:00"}
            },
            "nikolaeva_op": {
                "monday": {"start": "13:00", "end": "19:00"},
                "wednesday": {"start": "13:00", "end": "19:00"},
                "friday": {"start": "13:00", "end": "19:00"}
            }
        }
        
        self.data["schedule"] = schedule
        logger.info("Создано примерное расписание")
    
    def get_services_info(self, query: str = None) -> List[Dict[str, Any]]:
        """
        Получение информации об услугах.
        
        Args:
            query: Поисковый запрос (необязательно)
            
        Returns:
            Список услуг
        """
        try:
            services = self.data.get("services", [])
            
            if not query:
                return services
            
            # Простой поиск по запросу
            query_lower = query.lower()
            filtered_services = []
            
            for service in services:
                if (query_lower in service.get("name", "").lower() or
                    query_lower in service.get("category", "").lower() or
                    query_lower in service.get("description", "").lower()):
                    filtered_services.append(service)
            
            return filtered_services
            
        except Exception as e:
            logger.error(f"Ошибка получения услуг: {e}")
            return []
    
    def get_doctors_by_specialty(self, specialty: str) -> List[Dict[str, Any]]:
        """
        Получение врачей по специальности.
        
        Args:
            specialty: Специальность врача
            
        Returns:
            Список врачей
        """
        try:
            doctors = self.data.get("doctors", [])
            
            specialty_doctors = [
                doctor for doctor in doctors 
                if doctor.get("specialty") == specialty
            ]
            
            return specialty_doctors
            
        except Exception as e:
            logger.error(f"Ошибка получения врачей: {e}")
            return []
    
    def get_doctor_schedule(self, doctor_id: str, date: str = None) -> Dict[str, Any]:
        """
        Получение расписания врача.
        
        Args:
            doctor_id: ID врача
            date: Дата (если не указана, возвращается общее расписание)
            
        Returns:
            Расписание врача
        """
        try:
            schedule = self.data.get("schedule", {})
            doctor_schedule = schedule.get(doctor_id, {})
            
            if date:
                # Определяем день недели
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                weekday = date_obj.strftime("%A").lower()
                
                weekday_map = {
                    "monday": "monday",
                    "tuesday": "tuesday", 
                    "wednesday": "wednesday",
                    "thursday": "thursday",
                    "friday": "friday",
                    "saturday": "saturday",
                    "sunday": "sunday"
                }
                
                day_schedule = doctor_schedule.get(weekday_map.get(weekday), {})
                return {"date": date, "schedule": day_schedule}
            
            return {"doctor_id": doctor_id, "schedule": doctor_schedule}
            
        except Exception as e:
            logger.error(f"Ошибка получения расписания: {e}")
            return {}
    
    def check_appointment_availability(self, doctor_id: str, date: str, time: str) -> bool:
        """
        Проверка доступности времени для записи.
        
        Args:
            doctor_id: ID врача
            date: Дата
            time: Время
            
        Returns:
            True если время доступно
        """
        try:
            # Проверяем расписание врача
            schedule_info = self.get_doctor_schedule(doctor_id, date)
            day_schedule = schedule_info.get("schedule", {})
            
            if not day_schedule:
                return False  # Врач не работает в этот день
            
            # Проверяем, что время входит в рабочие часы
            start_time = day_schedule.get("start")
            end_time = day_schedule.get("end")
            
            if not start_time or not end_time:
                return False
            
            time_obj = datetime.strptime(time, "%H:%M").time()
            start_obj = datetime.strptime(start_time, "%H:%M").time()
            end_obj = datetime.strptime(end_time, "%H:%M").time()
            
            if not (start_obj <= time_obj <= end_obj):
                return False
            
            # Проверяем, что время не занято
            appointments = self.data.get("appointments", [])
            
            for appointment in appointments:
                if (appointment.get("doctor_id") == doctor_id and
                    appointment.get("date") == date and
                    appointment.get("time") == time and
                    appointment.get("status") != "cancelled"):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка проверки доступности: {e}")
            return False
    
    def create_appointment(self, appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создание записи на прием.
        
        Args:
            appointment_data: Данные записи
            
        Returns:
            Результат создания записи
        """
        try:
            # Генерируем ID записи
            appointment_id = f"apt_{len(self.data.get('appointments', []))+ 1:06d}"
            
            # Проверяем доступность
            is_available = self.check_appointment_availability(
                appointment_data.get("doctor_id"),
                appointment_data.get("date"),
                appointment_data.get("time")
            )
            
            if not is_available:
                return {
                    "success": False,
                    "message": "Выбранное время недоступно"
                }
            
            # Создаем запись
            appointment = {
                "id": appointment_id,
                "doctor_id": appointment_data.get("doctor_id"),
                "patient_name": appointment_data.get("patient_name"),
                "patient_phone": appointment_data.get("patient_phone"),
                "date": appointment_data.get("date"),
                "time": appointment_data.get("time"),
                "service_id": appointment_data.get("service_id"),
                "complaint": appointment_data.get("complaint", ""),
                "status": "scheduled",
                "created_at": datetime.now().isoformat()
            }
            
            # Добавляем в базу
            if "appointments" not in self.data:
                self.data["appointments"] = []
            
            self.data["appointments"].append(appointment)
            self._save_data()
            
            return {
                "success": True,
                "appointment_id": appointment_id,
                "message": f"Запись создана на {appointment_data.get('date')} в {appointment_data.get('time')}"
            }
            
        except Exception as e:
            logger.error(f"Ошибка создания записи: {e}")
            return {
                "success": False,
                "message": "Ошибка при создании записи"
            }
    
    def get_available_times(self, doctor_id: str, date: str) -> List[str]:
        """
        Получение доступного времени для записи.
        
        Args:
            doctor_id: ID врача
            date: Дата
            
        Returns:
            Список доступного времени
        """
        try:
            # Получаем расписание врача
            schedule_info = self.get_doctor_schedule(doctor_id, date)
            day_schedule = schedule_info.get("schedule", {})
            
            if not day_schedule:
                return []
            
            start_time = day_schedule.get("start")
            end_time = day_schedule.get("end")
            
            if not start_time or not end_time:
                return []
            
            # Генерируем временные слоты (каждые 30 минут)
            start_obj = datetime.strptime(start_time, "%H:%M")
            end_obj = datetime.strptime(end_time, "%H:%M")
            
            available_times = []
            current_time = start_obj
            
            while current_time < end_obj:
                time_str = current_time.strftime("%H:%M")
                
                # Проверяем доступность
                if self.check_appointment_availability(doctor_id, date, time_str):
                    available_times.append(time_str)
                
                current_time += timedelta(minutes=30)
            
            return available_times
            
        except Exception as e:
            logger.error(f"Ошибка получения доступного времени: {e}")
            return []
    
    def search_doctors(self, query: str) -> List[Dict[str, Any]]:
        """
        Поиск врачей по запросу.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Список врачей
        """
        try:
            doctors = self.data.get("doctors", [])
            query_lower = query.lower()
            
            filtered_doctors = []
            
            for doctor in doctors:
                if (query_lower in doctor.get("name", "").lower() or
                    query_lower in doctor.get("specialty", "").lower() or
                    query_lower in doctor.get("position", "").lower()):
                    filtered_doctors.append(doctor)
            
            return filtered_doctors
            
        except Exception as e:
            logger.error(f"Ошибка поиска врачей: {e}")
            return []
    
    def get_service_by_id(self, service_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение услуги по ID.
        
        Args:
            service_id: ID услуги
            
        Returns:
            Информация об услуге
        """
        try:
            services = self.data.get("services", [])
            
            for service in services:
                if service.get("id") == service_id:
                    return service
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка получения услуги: {e}")
            return None
    
    def get_doctor_by_id(self, doctor_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение врача по ID.
        
        Args:
            doctor_id: ID врача
            
        Returns:
            Информация о враче
        """
        try:
            doctors = self.data.get("doctors", [])
            
            for doctor in doctors:
                if doctor.get("id") == doctor_id:
                    return doctor
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка получения врача: {e}")
            return None
    
    def get_appointments_by_date(self, date: str) -> List[Dict[str, Any]]:
        """
        Получение записей на определенную дату.
        
        Args:
            date: Дата в формате YYYY-MM-DD
            
        Returns:
            Список записей
        """
        try:
            appointments = self.data.get("appointments", [])
            
            date_appointments = [
                apt for apt in appointments 
                if apt.get("date") == date and apt.get("status") != "cancelled"
            ]
            
            return date_appointments
            
        except Exception as e:
            logger.error(f"Ошибка получения записей: {e}")
            return []
    
    def cancel_appointment(self, appointment_id: str) -> Dict[str, Any]:
        """
        Отмена записи на прием.
        
        Args:
            appointment_id: ID записи
            
        Returns:
            Результат отмены
        """
        try:
            appointments = self.data.get("appointments", [])
            
            for appointment in appointments:
                if appointment.get("id") == appointment_id:
                    appointment["status"] = "cancelled"
                    appointment["cancelled_at"] = datetime.now().isoformat()
                    
                    self._save_data()
                    
                    return {
                        "success": True,
                        "message": "Запись успешно отменена"
                    }
            
            return {
                "success": False,
                "message": "Запись не найдена"
            }
            
        except Exception as e:
            logger.error(f"Ошибка отмены записи: {e}")
            return {
                "success": False,
                "message": "Ошибка при отмене записи"
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики медицинского центра.
        
        Returns:
            Статистическая информация
        """
        try:
            stats = {
                "total_services": len(self.data.get("services", [])),
                "total_doctors": len(self.data.get("doctors", [])),
                "total_appointments": len(self.data.get("appointments", [])),
                "active_appointments": len([
                    apt for apt in self.data.get("appointments", [])
                    if apt.get("status") == "scheduled"
                ]),
                "cancelled_appointments": len([
                    apt for apt in self.data.get("appointments", [])
                    if apt.get("status") == "cancelled"
                ])
            }
            
            # Статистика по специальностям
            doctors = self.data.get("doctors", [])
            specialties = {}
            
            for doctor in doctors:
                specialty = doctor.get("specialty", "unknown")
                if specialty in specialties:
                    specialties[specialty] += 1
                else:
                    specialties[specialty] = 1
            
            stats["doctors_by_specialty"] = specialties
            
            return stats
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {"error": str(e)}
    
    def add_patient(self, patient_data: Dict[str, Any]) -> str:
        """
        Добавление пациента в базу.
        
        Args:
            patient_data: Данные пациента
            
        Returns:
            ID пациента
        """
        try:
            # Генерируем ID пациента
            patient_id = f"pat_{len(self.data.get('patients', [])) + 1:06d}"
            
            patient = {
                "id": patient_id,
                "name": patient_data.get("name"),
                "phone": patient_data.get("phone"),
                "email": patient_data.get("email", ""),
                "birth_date": patient_data.get("birth_date", ""),
                "created_at": datetime.now().isoformat()
            }
            
            if "patients" not in self.data:
                self.data["patients"] = []
            
            self.data["patients"].append(patient)
            self._save_data()
            
            logger.info(f"Добавлен пациент {patient_id}")
            
            return patient_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления пациента: {e}")
            return ""
    
    def find_patient_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        """
        Поиск пациента по номеру телефона.
        
        Args:
            phone: Номер телефона
            
        Returns:
            Данные пациента или None
        """
        try:
            patients = self.data.get("patients", [])
            
            for patient in patients:
                if patient.get("phone") == phone:
                    return patient
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка поиска пациента: {e}")
            return None
    
    def close(self):
        """Закрытие сервиса и сохранение данных."""
        try:
            self._save_data()
            logger.info("Медицинская база данных закрыта")
        except Exception as e:
            logger.error(f"Ошибка закрытия базы данных: {e}")