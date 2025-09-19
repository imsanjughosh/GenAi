import nltk

# Make sure both punkt and punkt_tab are downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

def sentence_chunking(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# chunks = sentence_chunking(sample_text)
# for i, c in enumerate(chunks):
#     print(f"Chunk {i+1}:", c)
