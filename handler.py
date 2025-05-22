import runpod
import torch
import re
import nltk
from nltk.tokenize import sent_tokenize
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Constants
SIMILARITY_THRESHOLD = 0.6
MIN_SUMMARY_TOKENS = 20
MAX_SUMMARY_TOKENS = 150
MIN_COSINE_SIMILARITY = 0.5
SUMMARY_RATIO = 0.22

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_name = "nsi319/legal-pegasus"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

api_key = "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"
# --- Helper Functions ---
def clean_text(text):
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_chunks(sentences, chunk_size):
    chunks, current_chunk = [], []
    word_count = 0
    for sent in sentences:
        sent_words = len(sent.split())
        if word_count + sent_words <= chunk_size:
            current_chunk.append(sent)
            word_count += sent_words
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sent]
            word_count = sent_words
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def summarize_document(doc_text):
    cleaned_text = clean_text(doc_text)
    doc_sentences = sent_tokenize(cleaned_text)
    doc_length = len(cleaned_text.split())
    target_length = int(doc_length * SUMMARY_RATIO)
    chunk_size = 512

    chunks = split_into_chunks(doc_sentences, chunk_size)
    generated_summary = []
    current_length = 0

    for chunk in chunks:
        if current_length >= target_length:
            break

        max_len = int(SUMMARY_RATIO * len(chunk.split()))
        min_len = max(max_len - 5, MIN_SUMMARY_TOKENS)
        if min_len <= 0 or max_len <= 0:
            continue

        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=5,
                min_length=min_len,
                max_length=max_len,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=True
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        if summary:
            if not summary.endswith('.'):
                summary += '.'
            current_length += len(summary.split())
            generated_summary.append(summary)

    final_summary = ' '.join(OrderedDict.fromkeys(sent_tokenize(' '.join(generated_summary))))
    
    
    return final_summary

# --- RunPod Handler ---
def handler(job):
    try:
        job_input = job["input"]
        document_text = job_input.get("text", "")
        if not document_text:
            return {"error": "No 'text' field provided in input."}

        summary = summarize_document(document_text)
        return {"summary": summary}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
