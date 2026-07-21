import spacy
import re
from heapq import nlargest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
# Load Spacy Model
nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):
    if not text:
        return ''
    text = text.lower()
    # remove common filler words
    filler_words = r'\b(um|uh|you know|actually|like|so|basically|seriously|literally)\b'
    text = re.sub(filler_words, '', text)
    # collapse whitespace and normalize spaces
    return re.sub(r'\s+', ' ', text).strip()

def compute_tfidf(sentences):
    #each column is word while each row is a sentence and each cell has the TF-IDF score
    #transforms the text document into matrix
    tfidf_vectorizer=TfidfVectorizer()
    #scoring the occurence of each word in difference sentences  
    tfidf_matrix=tfidf_vectorizer.fit_transform(sentences)
    return tfidf_matrix

def extractive_summary(text, summary_ratio=0.3):
    # Preprocessing the Text
    text=preprocess_text(text)
    # Ensuring text is not empty
    if not text.strip():
        raise ValueError("Input text is empty or invalid after preprocessing.")

    # Process Text with Spacy
    doc = nlp(text)
    word_freq = {}
    pos_weight = {"NOUN": 2, "VERB": 1.5, "ADV": 1, "ADJ": 1}
    for word in doc:
        #removing the stop words and punctuation for checking it's pos(part of speech) weight
        if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
            if word.pos_ in pos_weight:
                word_freq[word.text] = word_freq.get(word.text, 0) + pos_weight[word.pos_]

    # Normalize Frequencies
    if not word_freq:
        raise ValueError("No valid words found for summary scoring.")
    max_freq = max(word_freq.values())
    word_freq = {word: freq / max_freq for word, freq in word_freq.items()}

    # Sentence Scoring
    sentences = list(doc.sents)
    sent_scores = {}
    for sentence in sentences:
        for word in sentence:
            if word.text in word_freq:
                sent_scores[sentence] = sent_scores.get(sentence, 0) + word_freq[word.text]

    # TF-IDF for Redundancy Check
    sents_text = [sent.text.strip() for sent in sentences if len(sent.text.strip()) > 0]
    tfidf_matrix = compute_tfidf(sents_text)
    # helps to reduce redundancy: cosine similarity = A.B / |A||B|
    cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = cosine_matrix[i][j]
            if similarity > 0.3:
                if sentences[i] in sent_scores:
                    sent_scores[sentences[i]] -= similarity
                if sentences[j] in sent_scores:
                    sent_scores[sentences[j]] -= similarity

    # Select Top Sentences
    select_len = max(1, int(len(sentences) * summary_ratio))
    top_sentences = nlargest(select_len, sent_scores, key=sent_scores.get)
    sentence_order = {sent: idx for idx, sent in enumerate(sentences)}
    top_sentences.sort(key=lambda sent: sentence_order[sent])
    return " ".join([sent.text for sent in top_sentences])
