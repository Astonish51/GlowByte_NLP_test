import json
import re
import gc

from .functions_LR import count_words, text2emb_bert, dataset_pipeline
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
from catboost import CatBoostClassifier, Pool
import random


def dataset(train, val, test, tokenizer, target_train, target_val, target_test, batch_size = 240):
  """
  Подготавливает данные для обучения и валидации модели машинного обучения.

  Параметры:
  - train, val, test: DataFrame, содержащий текстовые данные для обучающей, валидационной и тестовой выборок соответственно.
  - tokenizer: объект токенизатора, используемый для кодирования текстовых данных.
  - target_train, target_val, target_test: массив, содержащий метки классов для обучающей, валидационной и тестовой выборок соответственно.
  - batch_size: int, размер пакета данных (по умолчанию 240).

  Возвращает:
  - train_dataloader: DataLoader для обучающей выборки.
  - val_dataloader: DataLoader для валидационной выборки.
  - test_dataloader: DataLoader для тестовой выборки.
  """

  encoded_data_train = tokenizer.batch_encode_plus(
      list(train),
      add_special_tokens=True,
      return_attention_mask=True,
      padding=True,
      return_tensors='pt'
  )

  encoded_data_val = tokenizer.batch_encode_plus(
      list(val),
      add_special_tokens=True,
      return_attention_mask=True,
      padding=True,
      return_tensors='pt'
  )

  encoded_data_test = tokenizer.batch_encode_plus(
      list(test),
      add_special_tokens=True,
      return_attention_mask=True,
      padding=True,
      return_tensors='pt'
  )


  train_seq = encoded_data_train['input_ids']
  train_mask = encoded_data_train['attention_mask']
  train_y = torch.tensor(target_train)

  val_seq = encoded_data_val['input_ids']
  val_mask = encoded_data_val['attention_mask']
  val_y = torch.tensor(target_val)

  test_seq = encoded_data_test['input_ids']
  test_mask = encoded_data_test['attention_mask']
  test_y = torch.tensor(target_test)

  train_data = TensorDataset(train_seq, train_mask, train_y)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)

  val_data =  TensorDataset(val_seq, val_mask, val_y)
  val_sampler = SequentialSampler(val_data)
  val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size)

  test_data =  TensorDataset(test_seq, test_mask, test_y)
  test_sampler = SequentialSampler(test_data)
  test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = batch_size)

  return train_dataloader, val_dataloader, test_dataloader



def f1_score_func_m(preds, labels, average='binary'):
    """
    Вычисляет F1-меру для многоклассовой или бинарной классификации.

    Параметры:
    - preds: numpy array, прогнозы модели
    - labels: numpy array, истинные метки
    - average: строка, опциональный параметр, задающий тип усреднения (по умолчанию 'binary')

    Возвращает:
    - f1_score: float, F1-мера
    """

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    f1 = f1_score(labels_flat, preds_flat, average=average)
    
    return f1

def trainer(model, criterion, optimizer, scheduler, train_dataloader, device='cpu'):
    """
    Обучает модель на обучающей выборке.

    Параметры:
    - model: PyTorch модель.
    - criterion: функция потерь.
    - optimizer: оптимизатор.
    - scheduler: планировщик скорости обучения.
    - train_dataloader: DataLoader для обучающей выборки.
    - device: устройство для обучения (по умолчанию 'cpu').

    Возвращает:
    - avg_loss: среднюю потерю на обучающей выборке.
    - total_preds: предсказания модели для всей обучающей выборки.
    """
    
    model.train()
    total_loss, total_accuracy = 0, 0
    total_preds = []

    for step, batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
        batch = [r.to(device) for r in batch]
        labels = batch[2].to(device)
        output = {'input_ids': batch[0],'attention_mask' : batch[1]}

        preds = model(**output)
        preds = preds['logits']
        loss = criterion(preds, labels)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1,0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis = 0)

    return avg_loss, total_preds

def evaluate(model, criterion, val_dataloader, device='cpu'):
    """
    Оценивает модель на валидационной выборке.

    Параметры:
    - model: PyTorch модель.
    - criterion: функция потерь.
    - val_dataloader: DataLoader для валидационной выборки.
    - device: устройство для оценки (по умолчанию 'cpu').

    Возвращает:
    - avg_loss: среднюю потерю на валидационной выборке.
    - total_preds: предсказания модели для всей валидационной выборки.
    - total_labels: истинные метки классов для всей валидационной выборки.
    """

    model.eval()
    total_loss, total_accuracy = 0,0
    total_preds = []
    total_labels = []

    for step, batch in tqdm(enumerate(val_dataloader), total = len(val_dataloader)):
        batch = [t.to(device) for t in batch]
        output = {'input_ids': batch[0],'attention_mask' : batch[1]}
        labels = batch[2].to(device)

        with torch.no_grad():
            preds = model(**output)
            preds = preds['logits']
            loss = criterion(preds, labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            total_preds.append(preds)
            total_labels.append(labels)
    avg_loss = total_loss / len(val_dataloader)
    total_preds = np.concatenate(total_preds, axis = 0)
    total_labels = np.concatenate(total_labels, axis = 0)


    return avg_loss, total_preds, total_labels
    
def trainer_pipeline(PATH_MODEL, model, HF, optimizer, scheduler, criterion, train_dataloader, val_dataloader, epochs=100, device='cpu'):
    """
    Обучает модель и оценивает ее на валидационной выборке в течение нескольких эпох.

    Параметры:
    - PATH_MODEL: путь для сохранения лучших весов модели.
    - model: PyTorch модель.
    - HF: загрузчик данных, например, Hugging Face (не используется в функции).
    - optimizer: оптимизатор.
    - scheduler: планировщик шага обучения.
    - criterion: функция потерь.
    - train_dataloader: DataLoader для обучающей выборки.
    - val_dataloader: DataLoader для валидационной выборки.
    - epochs: количество эпох (по умолчанию 100).
    - device: устройство для обучения (по умолчанию 'cpu').

    Возвращает:
    - train_losses: список потерь на обучающей выборке для каждой эпохи.
    - valid_losses: список потерь на валидационной выборке для каждой эпохи.
    - f_scores: список метрик F1 на валидационной выборке для каждой эпохи.
    """
    model.to(device)

    # заморозим все параметры предобученой модели кроме последнего блока внимания и слоев обощающего и классифицирующего над ним
    for n,p in model.named_parameters():
      if 'classifier' in n or 'pooler' in n or '11' in n:
        p.requires_grad = True
      else:
        p.requires_grad = False

    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    f_scores = []

    for epoch in range(epochs):
        print('\n Epoch{:} / {:}'.format(epoch+1, epochs))

        train_loss, _ = trainer(model, criterion, optimizer, scheduler, train_dataloader, device=device)
        valid_loss, valid_pred, valid_labels = evaluate(model, criterion, val_dataloader, device=device)

        clear_output(True)

        if epoch % 10 == 0:
          print(classification_report(valid_labels, np.argmax(valid_pred, axis=1).flatten(), zero_division=0))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), PATH_MODEL)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        score = f1_score_func_m(valid_pred, valid_labels, average='macro')
        f_scores.append(score)
        
        print(f'\nTraining loss: {train_loss:.3f}')
        print(f'Validation loss: {valid_loss:.3f}')
        print(score)  

        plt.plot(train_losses, label='train')
        plt.plot(valid_losses, label='valid')
        plt.plot(f_scores, label='f_score')
        plt.legend()
        plt.show()

    return train_losses, valid_losses, f_scores

def inference(text, PATH_MODEL, pca, tfidf, model_lr, tokenizer, HF, lable2token, trsh=1, device='cpu'):
  """
  Функция для проведения инференса (предсказания).

  Параметры:
  - text: список строк, содержащих тексты для классификации
  - PATH_MODEL: строка, путь к сохраненным весам модели
  - pca: объект PCA для сокращения размерности данных (может быть None, если не используется)
  - tfidf: TF-IDF векторизатор для предобработки текста
  - model_lr: обученная модель логистической регрессии
  - tokenizer: токенизатор для предобработки текста
  - HF: строка, идентификатор предобученной модели (например, 'bert-base-uncased')
  - lable2token: словарь сопоставления классов меток и их числовых значений
  - trsh: float, порог для смешивания предсказаний (по умолчанию 1)

  Возвращает:
  - predicted_class: массив с предсказанными числовыми значениями классов
  - class_name: массив с именами предсказанных классов
  """
  X_train, X_val, y_train, y_val, _, _ = dataset_pipeline(text,
                                                          0,
                                                          tokenizer_bert = AutoTokenizer.from_pretrained(HF, do_lower_case=True),
                                                          model_bert = AutoModel.from_pretrained(HF),
                                                          pca = pca,
                                                          tfidf = tfidf, 
                                                          bert=False,
                                                          cv=False,
                                                          is_test=True,
                                                          device=device)
  logit = model_lr
  y_pred = logit.predict(X_val)
  y_pred_proba = logit.predict_proba(X_val)

  model = AutoModelForSequenceClassification.from_pretrained(HF, num_labels=12, ignore_mismatched_sizes=True)
  model.load_state_dict(torch.load(PATH_MODEL, map_location=device))
  model.to(device)
  model.eval()
  inputs = tokenizer(text.to_list(), return_tensors='pt', truncation=True, padding=True).to(device)
  outputs = model(**inputs)
  logits = outputs.logits
  logist_softmax = F.softmax(torch.tensor(logits)).detach().cpu().numpy()
  predicted_class = np.argmax(logist_softmax*(1 - trsh) + y_pred_proba*trsh, axis=1)

  class_name = []
  for c in predicted_class:
    for i, j in lable2token.items():
      if j==c:
        class_name.append(i)
      else:
        assert 'class out of bound'

  return predicted_class, class_name