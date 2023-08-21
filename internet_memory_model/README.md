# Описание структуры

Модуль internet_memory_model пердназначен для обучения и использования модели, которая может получать информацию из сети Интернет и долговременной памяти:
 - fid.py - реализация метода Fusion-in-Decoder внутри архитектуры модели T5
 - in_dataset.py - предобработка датасетов и токенизация текста
 - in_model.py - архитектура основной модели
 - inference.py - реализация чата с обученной моделью
 - train_internet_model.py - обучение модели
 - test_internet_model.py - тестирование модели
 - utils.py - подсчет метрик, обработка текста, полученного через интернет-поиск

 - В папке retriever содержится код для обучения и тестирования ранжирующей модели:
   - dataset.py - файл для предобработки корпусов
   - model.py - файл с архитектурой модели
   - train.py - файл для обучения модели
   - test.py - файл для тестирования модели

# Установка модулей

```
pip install -U bitsandbytes
pip install -U git+https://github.com/huggingface/transformers.git 
pip install -U git+https://github.com/huggingface/peft.git
pip install -U git+https://github.com/huggingface/accelerate.git
```

# Описание API

Обучение основной модели:
```
python train_internet_model.py
```
Тестирование основной модели:
```
python test_internet_model.py
```
Чат с основной моделью:
```
python test_internet_model.py
```
Для ведения диалога с основной моделью необходима обученная ранжирующая модель:
```
cd retriever/
python train.py
```

# Метрики

Тестирование проводилось на датасете Toloka Persona Chat Rus

| Название модели | ppl | bleu-1 | bleu-2 |
|--- | --- | --- | --- |
| ai-forever/ruT5-base | 10,33 | 0,40 | 0,34 |
| ai-forever/ruT5-large | 8,86 | 0,40 | 0,35 |
| ai-forever/FRED-T5-1.7B | 4,26 | 0,42 | 0,36 |