import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")


def _extract_nouns(text):
    doc = nlp(text)
    return [token.text.lower() for token in doc if token.pos_ == "NOUN"]


def map_by_cosine_similarity(
    reviews: str, menue_items: list, score_threshold: float = 0.5
):

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    menu_vectors = vectorizer.fit_transform([item.lower() for item in menue_items])

    nouns = _extract_nouns(reviews)
    matches = set()
    for noun in nouns:
        noun_vector = vectorizer.transform([noun])
        if noun_vector.sum() == 0:
            continue
        similarities = cosine_similarity(noun_vector, menu_vectors)
        max_idx = similarities.argmax()
        max_score = similarities[0, max_idx]
        if max_score > score_threshold:
            matches.add(menue_items[max_idx])
    return list(matches)
