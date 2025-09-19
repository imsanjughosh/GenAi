def fixed_size_chunking(text):
    chunk_size=50
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# chunks = fixed_size_chunking(sample_text, chunk_size=50)
# for i, c in enumerate(chunks):
#     print(f"Chunk {i+1}:", c)
