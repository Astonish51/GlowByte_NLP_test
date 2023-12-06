import json
import re
import gc
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from scipy.sparse import coo_matrix, hstack

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.nn import functional as F
from tqdm.notebook import tqdm
from IPython.display import clear_output

from transformers import (
    BertTokenizer, BertForSequenceClassification, AdamW, 
    get_linear_schedule_with_warmup, DistilBertTokenizer, 
    AutoModelForSequenceClassification, DistilBertForSequenceClassification, 
    RobertaTokenizer, RobertaForSequenceClassification, AutoModel, 
    AutoTokenizer, Trainer, TrainingArguments
)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, multilabel_confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import catboost
from catboost import CatBoostClassifier


def count_words(df, top=30):
    """
    Функция для подсчета топ-30 частотно встречаемых слов в колонке DataFrame.

    Параметры:
    - df: Series, содержащий колонку с текстом
    - top: int, кол-во строк для отображения
    Возвращает:
    - Series с топом частотно встречаемых слов
    """

    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(df)

    feature_names_count = count_vectorizer.get_feature_names_out()
    df_count = pd.DataFrame(count_matrix.toarray(), columns=feature_names_count)

    word_counts = df_count.sum()
    top_words = word_counts.sort_values(ascending=False)[:top]

    return top_words


def text2emb_bert(x, model, tokenizer, device='cpu'):
    """
    Возвращает векторное представление текста, полученное с использованием модели BERT.

    Параметры:
    - x: str, текст для векторизации
    - model: объект модели BERT
    - tokenizer: объект токенизатора BERT

    Возвращает:
    - numpy array, векторное представление текста
    """
    tokenized = tokenizer.encode(x)
    input_ids = torch.tensor([tokenized]).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    output = {'input_ids': input_ids, 'attention_mask': attention_mask}
    preds = model(**output)

    embeddings = F.normalize(preds['pooler_output'], p=2, dim=1).detach().cpu().numpy()[0]

    return embeddings


def dataset_pipeline(data, target, tokenizer_bert, model_bert, pca, tfidf, index_train=None, index_val=None, bert=False, cv=True, is_test=False, device='cpu'):
    """
    Пайплайн для подготовки данных для обучения и валидации модели машинного обучения.

    Параметры:
    - data: DataFrame, содержащий текстовые данные
    - target: массив, содержащий метки классов
    - tokenizer_bert: токенизатор BERT
    - model_bert: модель BERT
    - pca: метод сокращения размерности (PCA)
    - tfidf: TF-IDF векторизатор
    - scaler: стандартизатор
    - index_train: индексы обучающей выборки
    - index_val: индексы валидационной выборки
    - bert: флаг использования BERT (True, если используется)
    - cv: флаг использования кросс-валидации (True, если используется)
    - is_test: флаг, указывающий, является ли выборка тестовой
    - device: устройство для вычислений (cpu или cuda)

    Возвращает:
    - X_train, X_val: массивы с подготовленными данными для обучения и валидации
    - y_train, y_val: массивы с соответствующими метками классов
    - pca: обученный PCA
    - tfidf: обученный TF-IDF векторизатор
    """

    # если используем CV на обучающей выборке
    if cv:
        if is_test:
            assert f'test с CV не работает'
        data_train, data_val, y_train, y_val = data[index_train], data[index_val], target[index_train], target[index_val]

    # если используем валидационную выборку 
    elif not cv and not is_test:
        data_train, data_val, y_train, y_val = data[0].values, data[1].values, target[0], target[1]

    # если используем тестовую выборку 
    elif not cv and is_test:
        data_train, data_val, y_train, y_val = data.values, data.values, [0]*len(data), [0]*len(data)

    stratify = y_val
    
    if not is_test:
      data_train_tfidf = tfidf.fit_transform(data_train)
    else:
      data_train_tfidf = tfidf.transform(data_train)
    data_val_tfidf = tfidf.transform(data_val)

    if bert:
        data_train_emb = [text2emb_bert(x, model_bert, tokenizer_bert, device=device) for x in data_train]
        data_val_emb = [text2emb_bert(x, model_bert, tokenizer_bert, device=device) for x in data_val]

        data_train_emb = coo_matrix(data_train_emb)
        data_val_emb = coo_matrix(data_val_emb)

        X_train = hstack([data_train_emb, data_train_tfidf]).toarray()
        X_val = hstack([data_val_emb, data_val_tfidf]).toarray()

    else:
        X_train = data_train_tfidf.toarray()
        X_val = data_val_tfidf.toarray()
    
    if not is_test:
      X_train = pca.fit_transform(X_train)
    else:
      X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)

    gc.collect()
    torch.cuda.empty_cache()

    return X_train, X_val, y_train, y_val, pca, tfidf
  
