"""
RAG (Retrieval-Augmented Generation) сервис для медицинской базы знаний.
"""
import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredFileLoader

logger = logging.getLogger(__name__)

class RAGService:
    """Сервис для работы с медицинской базой знаний через RAG."""
    
    def __init__(self, openai_api_key: str, knowledge_base_path: str = None):
        """
        Инициализация RAG сервиса.
        
        Args:
            openai_api_key: API ключ OpenAI для эмбеддингов
            knowledge_base_path: Путь к папке с медицинскими документами
        """
        self.openai_api_key = openai_api_key
        self.knowledge_base_path = knowledge_base_path or self._get_default_knowledge_path()
        
        # Инициализируем компоненты
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = None
        
        self._setup_components()
        
    def _get_default_knowledge_path(self) -> str:
        """Получить путь к базе знаний по умолчанию."""
        current_dir = Path(__file__).resolve().parent.parent
        knowledge_path = current_dir / "data" / "medical_knowledge"
        knowledge_path.mkdir(parents=True, exist_ok=True)
        return str(knowledge_path)
    
    def _setup_components(self):
        """Настройка компонентов RAG."""
        try:
            # Инициализируем эмбеддинги
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key,
                model="text-embedding-ada-002"
            )
            
            # Настраиваем разделитель текста
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            # Инициализируем векторное хранилище
            self._setup_vectorstore()
            
            logger.info("RAG компоненты успешно настроены")
            
        except Exception as e:
            logger.error(f"Ошибка настройки RAG компонентов: {e}")
            raise
    
    def _setup_vectorstore(self):
        """Настройка векторного хранилища."""
        try:
            # Путь к базе данных Chroma
            persist_directory = os.path.join(self.knowledge_base_path, "chroma_db")
            
            # Проверяем, существует ли уже база данных
            if os.path.exists(persist_directory) and os.listdir(persist_directory):
                # Загружаем существующую базу
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Загружена существующая база знаний из {persist_directory}")
            else:
                # Создаем новую базу данных
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Создана новая база знаний в {persist_directory}")
                
                # Загружаем документы, если они есть
                self._load_initial_documents()
                
        except Exception as e:
            logger.error(f"Ошибка настройки векторного хранилища: {e}")
            raise
    
    def _load_initial_documents(self):
        """Загрузка начальных документов в базу знаний."""
        try:
            # Проверяем наличие документов
            documents_path = Path(self.knowledge_base_path)
            
            if not documents_path.exists():
                logger.warning(f"Папка с документами не найдена: {documents_path}")
                self._create_sample_documents()
                return
            
            # Загружаем все документы из папки
            documents = []
            
            for file_path in documents_path.glob("**/*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.md']:
                    try:
                        if file_path.suffix.lower() == '.pdf':
                            loader = PyPDFLoader(str(file_path))
                        else:
                            loader = TextLoader(str(file_path), encoding='utf-8')
                        
                        docs = loader.load()
                        
                        # Добавляем метаданные
                        for doc in docs:
                            doc.metadata.update({
                                'source': str(file_path),
                                'filename': file_path.name,
                                'file_type': file_path.suffix.lower()
                            })
                        
                        documents.extend(docs)
                        
                    except Exception as e:
                        logger.warning(f"Не удалось загрузить файл {file_path}: {e}")
            
            if documents:
                self.add_documents(documents)
                logger.info(f"Загружено {len(documents)} документов в базу знаний")
            else:
                logger.warning("Документы для загрузки не найдены")
                self._create_sample_documents()
                
        except Exception as e:
            logger.error(f"Ошибка загрузки начальных документов: {e}")
    
    def _create_sample_documents(self):
        """Создание примеров медицинских документов."""
        try:
            sample_docs = [
                {
                    "filename": "services.txt",
                    "content": """
УСЛУГИ МЕДИЦИНСКОГО ЦЕНТРА

ТЕРАПИЯ
- Первичная консультация терапевта - 2500 руб.
- Повторная консультация - 2000 руб.
- ЭКГ с расшифровкой - 800 руб.
- Измерение АД - 300 руб.

КАРДИОЛОГИЯ
- Консультация кардиолога - 3000 руб.
- ЭКГ + консультация - 3500 руб.
- УЗИ сердца (ЭхоКГ) - 2800 руб.
- Холтер мониторинг - 4500 руб.

НЕВРОЛОГИЯ
- Консультация невролога - 2800 руб.
- ЭЭГ - 3200 руб.
- УЗИ сосудов головного мозга - 2600 руб.

ГИНЕКОЛОГИЯ
- Консультация гинеколога - 2500 руб.
- УЗИ органов малого таза - 2200 руб.
- Кольпоскопия - 1800 руб.

УРОЛОГИЯ
- Консультация уролога - 2700 руб.
- УЗИ простаты - 2400 руб.
- Урофлоуметрия - 1500 руб.

ПЕДИАТРИЯ
- Консультация педиатра - 2200 руб.
- Осмотр новорожденного - 2800 руб.
- Профилактический осмотр - 1800 руб.

АНАЛИЗЫ
- Общий анализ крови - 500 руб.
- Биохимический анализ крови - 1200 руб.
- Общий анализ мочи - 400 руб.
- Анализ на инфекции - от 800 руб.
- Гормоны щитовидной железы - 1800 руб.
- Онкомаркеры - от 1500 руб.
"""
                },
                {
                    "filename": "doctors_schedule.txt",
                    "content": """
РАСПИСАНИЕ ВРАЧЕЙ

ТЕРАПЕВТЫ:
- Иванова Анна Петровна - Пн, Ср, Пт: 9:00-15:00
- Петров Сергей Михайлович - Вт, Чт: 10:00-16:00, Сб: 9:00-13:00

КАРДИОЛОГИ:
- Сидорова Елена Владимировна - Пн, Ср: 14:00-19:00
- Михайлов Игорь Константинович - Вт, Чт, Пт: 9:00-14:00

НЕВРОЛОГИ:
- Козлова Мария Александровна - Пн, Вт, Чт: 10:00-16:00
- Федоров Дмитрий Сергеевич - Ср, Пт: 14:00-20:00

ГИНЕКОЛОГИ:
- Романова Людмила Ивановна - Вт, Чт, Сб: 9:00-15:00
- Николаева Ольга Павловна - Пн, Ср, Пт: 13:00-19:00

УРОЛОГИ:
- Смирнов Алексей Викторович - Пн, Ср, Пт: 10:00-16:00
- Волков Владимир Николаевич - Вт, Чт: 14:00-19:00

ПЕДИАТРЫ:
- Морозова Екатерина Сергеевна - Пн-Пт: 9:00-15:00
- Зайцева Наталья Александровна - Сб, Вс: 10:00-16:00
"""
                },
                {
                    "filename": "medical_info.txt",
                    "content": """
МЕДИЦИНСКАЯ ИНФОРМАЦИЯ

ПОДГОТОВКА К АНАЛИЗАМ:
- Общий анализ крови: сдается натощак утром (8-12 часов голодания)
- Биохимический анализ: 12-часовое голодание, исключить алкоголь за 48 часов
- Анализ мочи: утренняя порция, средний поток, после туалета половых органов
- Гормоны щитовидной железы: утром натощак, исключить йодсодержащие препараты
- Онкомаркеры: утром натощак, женщинам в первую фазу цикла

СИМПТОМЫ, ТРЕБУЮЩИЕ СРОЧНОГО ОБРАЩЕНИЯ К ВРАЧУ:
- Острая боль в груди, особенно с отдачей в руку или челюсть
- Затрудненное дыхание, одышка в покое
- Высокая температура (выше 39°C) с ознобом
- Сильная головная боль с рвотой и нарушением зрения
- Потеря сознания или предобморочное состояние
- Сильное кровотечение (более 2-х прокладок в час)
- Острая боль в животе с напряжением мышц
- Судороги или нарушение речи

ПРОФИЛАКТИЧЕСКИЕ ОСМОТРЫ:
- Общий анализ крови и мочи - раз в год
- ЭКГ - раз в год после 40 лет, раз в 2 года до 40 лет
- Флюорография - раз в год
- Маммография - раз в 2 года после 40 лет
- Гинекологический осмотр - раз в год
- УЗИ органов брюшной полости - раз в 2 года
- Измерение АД - ежемесячно после 40 лет

РЕКОМЕНДАЦИИ ПО ЗДОРОВОМУ ОБРАЗУ ЖИЗНИ:
- Регулярные физические упражнения (150 минут умеренной активности в неделю)
- Сбалансированное питание: 5 порций овощей и фруктов в день
- Достаточный сон (7-8 часов для взрослых)
- Отказ от курения и ограничение алкоголя
- Управление стрессом (медитация, йога, хобби)
- Регулярные профилактические осмотры
- Поддержание нормального веса (ИМТ 18.5-24.9)
"""
                },
                {
                    "filename": "emergency_protocols.txt",
                    "content": """
ПРОТОКОЛЫ ЭКСТРЕННЫХ СИТУАЦИЙ

ПРИЗНАКИ ИНФАРКТА МИОКАРДА:
- Давящая, жгучая боль за грудиной
- Боль отдает в левую руку, челюсть, шею
- Тошнота, рвота, холодный пот
- Одышка, слабость
ДЕЙСТВИЕ: Немедленно вызвать скорую 103, принять нитроглицерин

ПРИЗНАКИ ИНСУЛЬТА:
- Внезапная слабость в руке или ноге
- Нарушение речи или понимания
- Асимметрия лица
- Головокружение, потеря координации
ДЕЙСТВИЕ: Немедленно вызвать скорую 103, не давать лекарств

АЛЛЕРГИЧЕСКИЕ РЕАКЦИИ:
- Крапивница, отек кожи
- Затрудненное дыхание
- Отек лица, губ, языка
- Падение АД, потеря сознания
ДЕЙСТВИЕ: Исключить аллерген, принять антигистаминное, при тяжелых симптомах - скорая

ВЫСОКАЯ ТЕМПЕРАТУРА:
- Выше 39°C у взрослых
- Выше 38.5°C у детей
- Судороги на фоне температуры
- Нарушение сознания
ДЕЙСТВИЕ: Физическое охлаждение, жаропонижающие, обильное питье

ТРАВМЫ:
- Переломы: обездвижить, не перемещать без необходимости
- Кровотечения: прямое давление, жгут при артериальном
- Ожоги: охладить водой, не вскрывать пузыри
- Отравления: вызвать рвоту (кроме кислот и щелочей)
"""
                },
                {
                    "filename": "common_symptoms.txt",
                    "content": """
ЧАСТО ВСТРЕЧАЮЩИЕСЯ СИМПТОМЫ И РЕКОМЕНДАЦИИ

ГОЛОВНАЯ БОЛЬ:
Возможные причины:
- Напряжение, стресс
- Мигрень
- Повышенное/пониженное АД
- Остеохондроз шейного отдела
К какому врачу: терапевт, невролог

БОЛИ В СПИНЕ:
Возможные причины:
- Остеохондроз
- Мышечное напряжение
- Межпозвоночная грыжа
- Радикулит
К какому врачу: терапевт, невролог, ортопед

БОЛИ В ЖИВОТЕ:
Возможные причины:
- Гастрит, язва
- Аппендицит
- Панкреатит
- Кишечные инфекции
К какому врачу: терапевт, гастроэнтеролог

КАШЕЛЬ:
Возможные причины:
- ОРВИ, грипп
- Бронхит
- Пневмония
- Аллергия
К какому врачу: терапевт, пульмонолог

ОДЫШКА:
Возможные причины:
- Сердечная недостаточность
- Бронхиальная астма
- Анемия
- Ожирение
К какому врачу: терапевт, кардиолог, пульмонолог

СЕРДЦЕБИЕНИЕ:
Возможные причины:
- Аритмия
- Стресс, тревога
- Гипертиреоз
- Анемия
К какому врачу: терапевт, кардиолог

ГОЛОВОКРУЖЕНИЕ:
Возможные причины:
- Вестибулярные нарушения
- Анемия
- Гипотония
- Остеохондроз
К какому врачу: терапевт, невролог, ЛОР

ОБЩИЕ РЕКОМЕНДАЦИИ:
- При появлении симптомов не откладывайте визит к врачу
- Ведите дневник симптомов
- Не занимайтесь самолечением
- При острых симптомах обращайтесь немедленно
"""
                }
            ]
            
            # Создаем файлы
            for doc_info in sample_docs:
                file_path = Path(self.knowledge_base_path) / doc_info["filename"]
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(doc_info["content"])
            
            logger.info("Созданы примеры медицинских документов")
            
            # Загружаем созданные документы
            self._load_initial_documents()
            
        except Exception as e:
            logger.error(f"Ошибка создания примеров документов: {e}")
    
    def add_documents(self, documents: List[Document]):
        """
        Добавление документов в базу знаний.
        
        Args:
            documents: Список документов для добавления
        """
        try:
            if not documents:
                logger.warning("Нет документов для добавления")
                return
            
            # Разделяем документы на части
            split_docs = self.text_splitter.split_documents(documents)
            
            # Добавляем в векторное хранилище
            self.vectorstore.add_documents(split_docs)
            
            # Сохраняем изменения
            self.vectorstore.persist()
            
            logger.info(f"Добавлено {len(split_docs)} частей документов в базу знаний")
            
        except Exception as e:
            logger.error(f"Ошибка добавления документов: {e}")
            raise
    
    def search_medical_knowledge(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Поиск релевантной медицинской информации.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов для возврата
            
        Returns:
            Список релевантных документов
        """
        try:
            if not self.vectorstore:
                logger.error("Векторное хранилище не инициализировано")
                return []
            
            # Поиск по сходству
            results = self.vectorstore.similarity_search(
                query=query,
                k=top_k
            )
            
            logger.info(f"Найдено {len(results)} релевантных документов для запроса: {query[:50]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска в базе знаний: {e}")
            return []
    
    def search_with_scores(self, query: str, top_k: int = 5, score_threshold: float = 0.7) -> List[tuple]:
        """
        Поиск с оценками релевантности.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            score_threshold: Минимальная оценка релевантности
            
        Returns:
            Список кортежей (документ, оценка)
        """
        try:
            if not self.vectorstore:
                return []
            
            # Поиск с оценками
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            # Фильтруем по порогу оценки
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= score_threshold
            ]
            
            logger.info(f"Найдено {len(filtered_results)} релевантных документов (порог: {score_threshold})")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Ошибка поиска с оценками: {e}")
            return []
    
    def search_by_category(self, query: str, category: str = None) -> List[Document]:
        """
        Поиск по категории документов.
        
        Args:
            query: Поисковый запрос
            category: Категория (services, doctors, medical_info, emergency, symptoms)
            
        Returns:
            Список релевантных документов
        """
        try:
            if not category:
                return self.search_medical_knowledge(query)
            
            # Поиск с фильтром по метаданным
            results = self.vectorstore.similarity_search(
                query=query,
                k=10,
                filter={"filename": {"$regex": f".*{category}.*"}}
            )
            
            logger.info(f"Найдено {len(results)} документов в категории {category}")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска по категории: {e}")
            return self.search_medical_knowledge(query)
    
    def add_text_document(self, text: str, metadata: Dict[str, Any] = None):
        """
        Добавление текстового документа.
        
        Args:
            text: Текст документа
            metadata: Метаданные документа
        """
        try:
            # Создаем документ
            doc = Document(
                page_content=text,
                metadata=metadata or {}
            )
            
            # Добавляем в базу
            self.add_documents([doc])
            
            logger.info("Текстовый документ добавлен в базу знаний")
            
        except Exception as e:
            logger.error(f"Ошибка добавления текстового документа: {e}")
            raise
    
    def update_knowledge_base(self, documents_path: str = None):
        """
        Обновление базы знаний из папки с документами.
        
        Args:
            documents_path: Путь к папке с документами
        """
        try:
            path = documents_path or self.knowledge_base_path
            
            # Очищаем существующую базу
            self._clear_vectorstore()
            
            # Загружаем документы заново
            self.knowledge_base_path = path
            self._load_initial_documents()
            
            logger.info("База знаний обновлена")
            
        except Exception as e:
            logger.error(f"Ошибка обновления базы знаний: {e}")
            raise
    
    def _clear_vectorstore(self):
        """Очистка векторного хранилища."""
        try:
            # Удаляем все документы
            if self.vectorstore:
                self.vectorstore.delete_collection()
            
            # Пересоздаем хранилище
            self._setup_vectorstore()
            
            logger.info("Векторное хранилище очищено")
            
        except Exception as e:
            logger.error(f"Ошибка очистки векторного хранилища: {e}")
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        Получение статистики базы знаний.
        
        Returns:
            Словарь со статистикой
        """
        try:
            stats = {
                "knowledge_base_path": self.knowledge_base_path,
                "vectorstore_initialized": self.vectorstore is not None,
                "embeddings_model": "text-embedding-ada-002"
            }
            
            # Подсчитываем количество документов в папке
            path = Path(self.knowledge_base_path)
            if path.exists():
                files = list(path.glob("**/*"))
                stats["files_in_directory"] = len([f for f in files if f.is_file()])
            
            # Информация о векторной базе
            if self.vectorstore:
                try:
                    # Пытаемся получить количество документов в векторной базе
                    collection = self.vectorstore._collection
                    stats["documents_in_vectorstore"] = collection.count()
                except:
                    stats["documents_in_vectorstore"] = "unknown"
            
            return stats
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Закрытие соединений и освобождение ресурсов."""
        try:
            if self.vectorstore:
                # Сохраняем изменения перед закрытием
                self.vectorstore.persist()
            
            logger.info("RAG сервис закрыт")
            
        except Exception as e:
            logger.error(f"Ошибка закрытия RAG сервиса: {e}")