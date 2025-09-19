from Paragraph_level_chunking import paragraph_chunking
from Fixed_size_chunking import fixed_size_chunking

def recursive_chunking(text):
    max_chars=60
    paragraphs = paragraph_chunking(text)
    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            chunks.extend(fixed_size_chunking(para))
    return chunks

# chunks = recursive_chunking(sample_text, max_chars=60)
# for i, c in enumerate(chunks):
#     print(f"Chunk {i+1}:", c)
