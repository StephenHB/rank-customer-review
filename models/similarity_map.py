import spacy
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from textdistance import levenshtein
from functools import lru_cache

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("downloading...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

N_GRAM_SIZE = 3

def _extract_nouns(text):
    doc = nlp(text)
    return [token.text.lower() for token in doc if token.pos_ == "NOUN"]


@lru_cache(maxsize=None)
def get_ngrams(text):
    return set([text[i:i+N_GRAM_SIZE] for i in range(len(text)-N_GRAM_SIZE+1)])

def ngram_edit_search(review, menue_items , score_threshold=0.8, ngram_threshold=0.3):
    nouns = _extract_nouns(review)
    matches = set()
    menu_lower = [item.lower() for item in menue_items]
    menu_ngrams = {
    item: {item[i:i+N_GRAM_SIZE] for i in range(len(item) - N_GRAM_SIZE + 1)}
    for item in menu_lower}
    
    for noun in nouns:
        noun_ngrams = get_ngrams(noun)
        best_score = 0
        best_match = None
        
        # First filter by n-gram similarity
        for menu_item, menu_gram in menu_ngrams.items():
            intersection = len(noun_ngrams & menu_gram)
            union = len(noun_ngrams | menu_gram)
            if union == 0:
                continue
                
            jaccard = intersection / union
            if jaccard > ngram_threshold:
                # Then calculate edit distance for filtered candidates
                score = levenshtein.normalized_similarity(noun, menu_item)
                if score > best_score and score > score_threshold:
                    best_score = score
                    best_match = menu_item
        
        if best_match:
            original_case = menue_items[menu_lower.index(best_match)]
            matches.add(original_case)
    
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


def st_map(reviews: str, menue_items: list, score_threshold: float = 0.5):
    # The difference vs true BERT method
    # What is all-MiniLM-L6-v2 embedding model? is it BERT based?
    # Which Tokinizer was used? (good tokinizer selection)

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

# Define a n-gram search to match the menu items
# e.g. str eddit distance

# Senitity check -> multi frequency check
# Self label the data -> validate by the business_id

# predict dish_name(bool) by masking the pre and post words
