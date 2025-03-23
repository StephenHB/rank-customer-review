import re
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import spacy
import subprocess
from nltk.stem import WordNetLemmatizer

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("downloading...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def get_action_verbs(df, col_name, top_n_verbs=30):
    text_data_list = df[col_name].tolist()
    verbs = []
    for doc in nlp.pipe(text_data_list):
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "dobj"):
                verbs.append(token.lemma_)
        action_verbs = set([v for v, cnt in Counter(verbs).most_common(top_n_verbs)])

    return action_verbs


def get_menu_items(df, col_name, top_n_menu_items=100):
    text_data_list = df[col_name].tolist()
    menu_candidates = []
    for doc in nlp.pipe(text_data_list):
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:
                menu_candidates.append(chunk.text.lower())
    menu_items = set([item for item, cnt in Counter(menu_candidates).most_common(100)])
    return menu_items


lemmatizer = WordNetLemmatizer()


def lemmatize_dish(dish):
    doc = nlp(dish)
    return " ".join([token.lemma_ for token in doc])


def extract_dishes_spacy(text, action_verbs):
    doc = nlp(text)
    dishes = []
    for token in doc:
        if token.text.lower() in action_verbs and token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ == "dobj":
                    dish = " ".join([w.lemma_ for w in child.subtree])
                    dishes.append(dish)

    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) >= 2:
            dish = " ".join([w.lemma_ for w in chunk])
            dishes.append(dish)

    return list(set(dishes))


def filter_dishes_spacy(texts, action_verbs, min_freq=2):
    all_dishes = []
    # stop_words = {"food", "service", "restaurant", "place"}
    stop_words = set(stopwords.words('english'))
    for text in texts:
        dishes = extract_dishes_spacy(text, action_verbs)
        filtered = [
            dish
            for dish in dishes
            if not any(word in dish.split() for word in stop_words)
        ]
        all_dishes.extend(filtered)
    counter = Counter(all_dishes)
    return [dish for dish, freq in counter.items() if freq >= min_freq]


def apply_spacy_to_df(
    df, action_verbs, menu_items, filter_validate_menu=False, min_freq=2
):
    # Extract raw dishes
    df["dishes_raw"] = df["text_clean"].apply(
        lambda text: extract_dishes_spacy(text, action_verbs)
    )

    # Lemmatize the extracted dishes
    df["dishes_lemmatized"] = df["dishes_raw"].apply(
        lambda dishes: [lemmatize_dish(dish) for dish in dishes]
    )
    # Filter dishes based on frequency
    all_texts = df["text_clean"].tolist()
    filtered_dishes = filter_dishes_spacy(all_texts, action_verbs, min_freq)

    # Add filtered dishes to the DataFrame
    df["filtered_dishes"] = df["dishes_lemmatized"].apply(
        lambda dishes: [dish for dish in dishes if dish in filtered_dishes]
    )

    # Optionally validate dishes against a menu
    if filter_validate_menu:
        df["valid_dishes"] = df["filtered_dishes"].apply(
            lambda dishes: [dish for dish in dishes if dish in menu_items]
        )

    return df
