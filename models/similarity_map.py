import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_md")

def _extract_nouns(text):
    doc = nlp(text)
    return [token.text.lower() for token in doc if token.pos_ == "NOUN"]


def tfidf_map(reviews: str, menue_items: list, score_threshold: float = 0.5):

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


def spacy_map(reviews: str, menue_items: list, score_threshold: float = 0.5):
    # Get vectors for menu items
    menu_vectors = [nlp(item.lower()).vector for item in menue_items]

    nouns = _extract_nouns(reviews)
    matches = set()
    for noun in nouns:
        token = nlp(noun)[0]
        if not token.has_vector:
            continue

        similarities = cosine_similarity([token.vector], menu_vectors)
        max_idx = similarities.argmax()
        max_score = similarities[0, max_idx]
        if max_score > score_threshold:
            matches.add(menue_items[max_idx])
    return list(matches)


def bert_map(reviews: str, menue_items: list, score_threshold: float = 0.5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    menu_embeddings = model.encode([item.lower() for item in menue_items])

    nouns = _extract_nouns(reviews)
    if not nouns:
        return []

    noun_embeddings = model.encode(nouns)
    similarities = cosine_similarity(noun_embeddings, menu_embeddings)
    matches = set()
    for i in range(len(nouns)):
        max_score_idx = similarities[i].argmax()
        if similarities[i][max_score_idx] > score_threshold:
            matches.add(menue_items[max_score_idx])
    return list(matches)
