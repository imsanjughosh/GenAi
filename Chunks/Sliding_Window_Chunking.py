def sliding_window_chunking(text):
    window_size=50
    overlap=10
    chunks = []
    start = 0
    while start < len(text):
        end = start + window_size
        chunks.append(text[start:end])
        start += window_size - overlap
    return chunks

# chunks = sliding_window_chunking(sample_text)
# for i, c in enumerate(chunks):
#     print(f"Chunk {i+1}:", c)
