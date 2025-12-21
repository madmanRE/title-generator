import re
from collections import Counter

import pandas as pd

from rapidfuzz import fuzz

from sklearn.feature_extraction.text import TfidfVectorizer

from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc

import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

segmenter = Segmenter()
ner_tagger = NewsNERTagger(NewsEmbedding())
STOP_WORDS = stopwords.words('english') + stopwords.words('russian')


def normalize_query(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^\w\s]", "", q)
    return q


def phrase_in_query(phrase, query, threshold=85):
    return fuzz.partial_ratio(phrase, query) >= threshold


def extract_geo(texts):
    geo_phrases = Counter()

    for t in texts:
        doc = Doc(t)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)

        for span in doc.spans:
            if span.type in {"LOC", "GPE"}:
                geo_phrases[span.text.lower()] += 1

    return geo_phrases.most_common(1)[0][0] if geo_phrases else None


def extract_phrases(texts):
    vectorizer = TfidfVectorizer(
        ngram_range=(2, 4),
        stop_words=STOP_WORDS,
        max_df=1,
        min_df=1
    )

    X = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()
    scores = X.mean(axis=0).A1

    phrases = [
        f for f, s in sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
        if len(f.split()) >= 2
    ]

    return phrases[:20]


def group_similar_phrases(phrases, threshold=85):
    groups = []

    for p in phrases:
        for g in groups:
            if fuzz.token_set_ratio(p, g[0]) >= threshold:
                g.append(p)
                break
        else:
            groups.append([p])

    return groups


def select_canonical(group, df):
    freq = Counter()

    for phrase in group:
        for _, row in df.iterrows():
            if phrase_in_query(phrase, row["Query"]):
                freq[phrase] += row["Volume"]

    if freq:
        return freq.most_common(1)[0][0]

    return max(group, key=lambda x: len(x.split()))


def extract_entities(df):
    texts = df["Query"].str.lower().tolist()
    phrases = extract_phrases(texts)

    groups = group_similar_phrases(phrases)

    entities = []
    synonyms = {}

    for g in groups:
        canon = select_canonical(g, df)
        entities.append(canon)
        synonyms[canon] = [x for x in g if x != canon]

    return entities, synonyms


def extract_commercial_phrases(entities, df):
    commercial = []

    for e in entities:
        score = 0
        for _, row in df.iterrows():
            if e in row["Query"].lower():
                score += row["Volume"]
        if score > df["Volume"].mean():
            commercial.append(e)

    return commercial


def select_head_query(df):
    return df.sort_values("Volume", ascending=False).iloc[0]["Query"].lower()


def build_roles(df: pd.DataFrame):
    df = df.copy()

    df["Query"] = df["Query"].astype(str).str.lower()

    df["Volume"] = (
        df["Volume"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", 0)
        .astype(int)
    )

    df["Query"] = df["Query"].apply(normalize_query)

    texts = df["Query"].tolist()

    head = select_head_query(df)
    geo = extract_geo(texts)
    entities, synonyms = extract_entities(df)
    commercial = extract_commercial_phrases(entities, df)

    mandatory = {head}

    if entities:
        mandatory.add(entities[0])

    if geo:
        mandatory.add(geo)

    optional = set(entities[1:]) | set(commercial) - mandatory

    return {
        "head_phrase": head,
        "mandatory_phrases": sorted(mandatory),
        "optional_phrases": sorted(optional),
        "synonyms": synonyms
    }
