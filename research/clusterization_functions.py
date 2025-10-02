import pandas as pd
import json
import numpy as np
import re
import string

import string
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import pymorphy3
from pymorphy3 import MorphAnalyzer
from pymystem3 import Mystem
from stop_words import get_stop_words
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from tqdm import tqdm

from typing import List

# data_path = '../data/'
# prepared_data = 'prepared_data/'

morph = MorphAnalyzer()
punctuation = set(string.punctuation + "«»…—–,()")

def lemmatize_text(text):
    """
    Лемматизация текста с pymorphy3, фильтрация пустых токенов и знаков препинания
    Возвращает список лемм в нижнем регистре
    """
    if not isinstance(text, str):
        return []

    # токенизация: оставляем только слова (буквы)
    tokens = re.findall(r'\b[А-Яа-яЁёA-Za-z]+\b', text)
    
    # лемматизация и фильтрация
    lemmas = []
    for token in tokens:
        lemma = morph.parse(token)[0].normal_form  # берём основную лемму
        lemma = lemma.lower().strip()
        if lemma and all(c not in punctuation for c in lemma):
            lemmas.append(lemma)
    return lemmas


def get_umap_embs(
    embs,
    data_path: str,
    emb_type: str,
    emb_class: str,
    n_neighbors = 25,
    n_components = 128
):
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components, 
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    embeddings_umap = umap_model.fit_transform(embs)

    save_path = data_path + emb_type + "_embs_" + emb_class + ".npy"
    print(f'UMAP Save Path: {save_path}')

    np.save(save_path, embeddings_umap)

    return embeddings_umap

def get_candidates(
    cluster_texts,
    russian_stopwords,
    embedding_model,
    data_path,
    ngram_rng = 2,
):
    save_path = data_path + 'candidates_embs' + ".npy"
    print(f'Candidates Save Path: {save_path}')

    vectorizer = CountVectorizer(
        tokenizer = lemmatize_text,
        ngram_range=(1,ngram_rng),      # униграммы и биграммы
        min_df=2,
        max_df=0.95,                
        stop_words=russian_stopwords    # убираем стоп-слова
    )
    vectorizer.fit(cluster_texts)
    candidates = vectorizer.get_feature_names_out()

    cand_embs = embedding_model.encode(candidates, convert_to_numpy=True, show_progress_bar=False)

    np.save(save_path, cand_embs)

    return cand_embs, candidates


def get_kmeans_clusters(
    text_type, # text, title
    class_df,
    embeddings_umap,
    n_clusters = 10,
):
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=30,
        max_iter=500,
        random_state=42,
        algorithm='elkan'
    )

    labels_km = kmeans_model.fit_predict(embeddings_umap)
    class_df[f'{text_type}_cluster_km_id'] = labels_km

    return class_df

def get_top(sims, candidates):
    top_idx = sims.argsort()[::-1][:10]
    return [
        {"keyword": candidates[i], "similarity": float(sims[i])}
        for i in top_idx
    ]

def get_cluster_names(
    class_df,
    text_type, # text, title
    cand_embs,
    candidates
):
    cluster_labels = f"{text_type}_cluster_km_id"
    cluster_summaries = {}

    for cluster_id in tqdm(class_df[cluster_labels].unique(), total = len(class_df[cluster_labels].unique())):

        cluster_texts = class_df[class_df[cluster_labels] == cluster_id]['text_clean_emb'].values
        cluster_titles = class_df[class_df[cluster_labels] == cluster_id]['title_emb'].values
        cluster_reviews = class_df[class_df[cluster_labels] == cluster_id]['review_emb'].values

        centroid_texts = cluster_texts.mean(axis=0)
        centroid_titles = cluster_titles.mean(axis=0)
        centroid_reviews = cluster_reviews.mean(axis=0)

        sims_texts = cosine_similarity([centroid_texts], cand_embs)[0]
        sims_titles = cosine_similarity([centroid_titles], cand_embs)[0]
        sims_reviews= cosine_similarity([centroid_reviews], cand_embs)[0]


        cluster_summaries[cluster_id] = {
            "text": get_top(sims_texts, candidates),
            "title": get_top(sims_titles, candidates),
            "review": get_top(sims_reviews, candidates),
        }

    return cluster_summaries

def save_cluster_summary_to_json(
    cluster_summary,
    text_type,
    path_to_save,
    filename
):
    save_path = f'{path_to_save}{text_type}_{filename}'

    data_dict = {str(k): v for k, v in cluster_summary.items()}

    # def make_serializable(obj):
    #     if isinstance(obj, np.ndarray):
    #         return obj.tolist()
    #     if isinstance(obj, np.generic):  # np.float32, np.int64 и т.д.
    #         return obj.item()
    #     if isinstance(obj, dict):
    #         return {k: make_serializable(v) for k, v in obj.items()}
    #     if isinstance(obj, list):
    #         return [make_serializable(i) for i in obj]
    #     return obj

    # cluster_summary_serializable = make_serializable(cluster_summary)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    print(f"Cluster Names Save Path: {save_path}")


def save_df_to_pickle(
    class_df: pd.DataFrame,
    path_to_save,
    class_name
):
    save_path = f'{path_to_save}{class_name}_class_df.pkl'
    
    print(f"Class DF Save Path: {save_path}")

    class_df.to_pickle(save_path)


def save_results_cl(
    class_df,
    text_cluster_summaries,
    title_cluster_summaries,
    path_to_save,
    class_name
):
    save_df_to_pickle(class_df, path_to_save, class_name)

    save_cluster_summary_to_json(text_cluster_summaries, 'text', path_to_save, 'cluster_summaries')
    save_cluster_summary_to_json(title_cluster_summaries, 'title', path_to_save, 'cluster_summaries')










# def get_hdbscan_clusters(
#     class_df,
#     embeddings_umap,
#     cluster_method = 'eom' # 'leaf'
# ):
#     hdbscan_model = HDBSCAN(
#         min_cluster_size=25,       # ловим маленькие кластеры
#         min_samples=20,            # плотность точки
#         metric='euclidean',           # для нормализованных эмбеддингов
#         cluster_selection_method=cluster_method,  # листья иерархии → много мелких кластеров
#         prediction_data=True,
#         allow_single_cluster=False
#     )

#     labels = hdbscan_model.fit_predict(embeddings_umap)

#     # Сколько кластеров получилось
#     print("Число кластеров (не включая шум):", len(set(labels)) - (1 if -1 in labels else 0))
#     print("Размеры кластеров:", np.bincount(labels[labels >= 0]))
#     print("Количество шумовых точек:", np.sum(labels == -1))

#     return class_df