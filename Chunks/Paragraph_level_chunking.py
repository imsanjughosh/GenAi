def paragraph_chunking(text):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs

# chunks = paragraph_chunking(sample_text)
# for i, c in enumerate(chunks):
#     print(f"Chunk {i+1}:", c)
