import nltk
import spacy
import tiktoken  # For token-level chunking

# Download NLTK punkt tokenizer
nltk.download('punkt', quiet=True)

class TextChunker:
    def __init__(self, text):
        self.text = text
        # Load spaCy model for semantic chunking
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please run: python -m spacy download en_core_web_sm")
            self.nlp = None

    # 1. Word-level chunking
    def word_chunking(self, chunk_size=5):
        words = self.text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    # 2. Sentence-level chunking
    def sentence_chunking(self):
        return nltk.sent_tokenize(self.text)

    # 3. Paragraph-level chunking
    def paragraph_chunking(self):
        return [p.strip() for p in self.text.split("\n") if p.strip()]

    # 4. Fixed-size chunking (characters)
    def fixed_size_chunking(self, chunk_size=50):
        return [self.text[i:i+chunk_size] for i in range(0, len(self.text), chunk_size)]

    # 5. Sliding window chunking
    def sliding_window_chunking(self, window_size=50, overlap=10):
        chunks = []
        start = 0
        while start < len(self.text):
            end = start + window_size
            chunks.append(self.text[start:end])
            start += window_size - overlap
        return chunks

    # 6. Semantic chunking
    def semantic_chunking(self):
        if not self.nlp:
            raise ValueError("spaCy model not loaded. Run: python -m spacy download en_core_web_sm")
        doc = self.nlp(self.text)
        return [sent.text.strip() for sent in doc.sents]

    # 7. Recursive chunking
    def recursive_chunking(self, max_chars=60):
        paragraphs = self.paragraph_chunking()
        chunks = []
        for para in paragraphs:
            if len(para) <= max_chars:
                chunks.append(para)
            else:
                chunks.extend([para[i:i+max_chars] for i in range(0, len(para), max_chars)])
        return chunks

    # 8. Token-based chunking
    def token_chunking(self, chunk_size=50, model="gpt-3.5-turbo"):
        """
        Splits text into chunks based on token count.
        model: model name for tokenization
        """
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(self.text)
        
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
        return chunks


# ------------------------------
# 🔹 Example usage:
# ------------------------------
sample_text = """
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
It has become an essential part of the technology industry. AI research is highly technical and specialized.
The core problems of AI include programming computers for certain traits such as knowledge, reasoning, problem-solving, perception, learning, and planning.
"""

chunker = TextChunker(sample_text)

print("TOKEN CHUNKS:", chunker.token_chunking(20))
