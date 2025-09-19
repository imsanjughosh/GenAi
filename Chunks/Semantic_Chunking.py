import spacy
nlp = spacy.load("en_core_web_sm")

def semantic_chunking(text):
    doc = nlp(text)
    chunks = [sent.text.strip() for sent in doc.sents]
    return chunks

# chunks = semantic_chunking(sample_text)
# for i, c in enumerate(chunks):
#     print(f"Chunk {i+1}:", c)
