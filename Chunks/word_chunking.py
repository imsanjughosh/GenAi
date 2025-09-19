def word_chunking(text):
    chunk_size=5
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# chunks = word_chunking(sample_text, chunk_size=5)
# for i, c in enumerate(chunks):
#     print(f"Chunk {i+1}:", c)
