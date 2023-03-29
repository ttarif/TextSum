import requests
from bs4 import BeautifulSoup
import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define functions for text preprocessing and summarization
def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.lemma_ != '-PRON-']
    return ' '.join(tokens)

def summarize_text(text, n_sentences=3):
    sentences = nltk.sent_tokenize(text)
    cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_sentences)
    
    sentence_similarity = cosine_similarity(tfidf_matrix)
    
    # TextRank algorithm
    ranks = np.zeros(len(sentences))
    for _ in range(10):
        for i, sentence in enumerate(cleaned_sentences):
            for j, _ in enumerate(cleaned_sentences):
                if i != j:
                    ranks[i] += sentence_similarity[i][j] / (np.sum(sentence_similarity[j]) + 1)
        ranks = ranks / np.max(ranks)
        
    # Latent Semantic Analysis (LSA) algorithm
    u, s, vt = np.linalg.svd(tfidf_matrix.toarray())
    sentence_scores = np.dot(vt[:n_sentences, :], np.diag(s)).tolist()
    top_sentences = np.argsort(sentence_scores, axis=1)[:, -n_sentences:].flatten()
    
    summary = [sentences[i] for i in sorted(list(set(ranks.argsort()[-n_sentences:]) | set(top_sentences)))]
    
    return ' '.join(summary)

# Example usage
url = 'Sample URL'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text()
summary = summarize_text(text, n_sentences=4)
print(summary)
