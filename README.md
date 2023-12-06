## Тестовое задание по классификации текстов

### Описание задачи

Необходимо разработать качественный классификатор текстов новостей.  

### Набор данных

Датасет содержит 835 текстов новостей (поле ‘content’) и 16 классов (поле ‘classification’) этих новостей.  
Данные сильно несбалансированны.  

### Решение  

Задача представляет собой многоклассовую классификацию текстов.  
В качестве решения используется комбинация предсказаний Logistic Regression с TF-IDF и BERT.  
Задание выполнено на языке Python.  
В качестве метрика используется f1-macro  
Решение в файле **GlowByte_NLP.ipynb**

### Требование к оборудованию

Для воспроизведения результатов приведенных в ноутбуке может потребоваться GPU.  
Также происходит дообучение BERT(https://huggingface.co/ProsusAI/finbert), что без GPU займет очень много времени, но в папке models есть педобученная модель.  
Используйте ее.  
Других специфичных требований к ПК нет.  

### Используемые библиотеки

Cписок библиотек и фреймворков, которые используются в проекте:

- PyTorch
- Scikit-learn
- NLTK
- CatBoost
- transformers
- scipy
- pandas
- numpy
- nlpaug - библиотека для аугументации текста https://github.com/makcedward/nlpaug

## Требования к подготовке среды

1. **Python 3.9+** : Убедитесь, что на вашем компьютере установлена версия Python 3.9+ [Официальный сайт Python](https://www.python.org/)

2. **Виртуальное окружение (рекомендуется)**: Рекомендуется создать виртуальное окружение для изоляции зависимостей.

   python -m venv venv
   source venv/bin/activate  # Для Linux / macOS  
   .\venv\Scripts\activate   # Для Windows  

   pip install -r requirements.txt  

   Для установки зависимостей в jupyter notebook используйте  
   ! pip install -r requirements.txt  
3. Импорт PyTorch (import torch) имеет ряд особенностей, лучше уточнить причину ошибок, если они возникли, на официальном сайте [Официальный сайт pytorch](https://pytorch.org/) или на [stackoverflow](https://stackoverflow.com/)

### Структура 

В **GlowByte_NLP.ipynb** предстален ход исследования и результаты.  
Функции, используемые в ноутбуке вынесены в два файла **functions_LR.py** и **functions_bert.py** в папке **src**.  
Веса обученых BERT лежат в **models**.  

├── data  
│      ├── GBC_NLP_test_news_sample.json  
├── models  
       ├── bert_multiclass1.pt  
       ├── bert_multiclass_balanced11.pt  
├── GlowByte_NLP.ipynb  
├── src  
│      ├── functions_LR.py  
│      ├── functions_bert.py  
├── README.md  
├── requirements.txt  
