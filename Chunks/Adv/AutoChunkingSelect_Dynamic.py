import os
import nltk
import spacy
import tiktoken

# Download NLTK punkt tokenizer
nltk.download('punkt', quiet=True)

class TextChunker:
    def __init__(self, text=None, file_path=None):
        """
        Initialize TextChunker with either direct text or a file path.
        If file_path is given, it will load the text from the file.
        """
        if file_path:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"❌ File not found: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                self.text = f.read()
        elif text:
            self.text = text
        else:
            raise ValueError("⚠️ Provide either 'text' or 'file_path'")

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please run: python -m spacy download en_core_web_sm")
            self.nlp = None

    def _make_metadata(self, chunk, start_idx, end_idx, token_count=None):
        return {
            "chunk_text": chunk,
            "start_index": start_idx,
            "end_index": end_idx,
            "token_count": token_count
        }

    # Word-level
    def word_chunking(self, chunk_size=5):
        words = self.text.split()
        chunks, idx = [], 0
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            start_idx = self.text.find(chunk, idx)
            end_idx = start_idx + len(chunk)
            idx = end_idx
            chunks.append(self._make_metadata(chunk, start_idx, end_idx))
        return chunks

    # Sentence-level
    def sentence_chunking(self):
        sentences = nltk.sent_tokenize(self.text)
        chunks, idx = [], 0
        for sent in sentences:
            start_idx = self.text.find(sent, idx)
            end_idx = start_idx + len(sent)
            idx = end_idx
            chunks.append(self._make_metadata(sent, start_idx, end_idx))
        return chunks

    # Paragraph-level
    def paragraph_chunking(self):
        paragraphs = [p.strip() for p in self.text.split("\n") if p.strip()]
        chunks, idx = [], 0
        for para in paragraphs:
            start_idx = self.text.find(para, idx)
            end_idx = start_idx + len(para)
            idx = end_idx
            chunks.append(self._make_metadata(para, start_idx, end_idx))
        return chunks

    # Fixed-size (characters)
    def fixed_size_chunking(self, chunk_size=50):
        return [
            self._make_metadata(self.text[i:i+chunk_size], i, min(i+chunk_size, len(self.text)))
            for i in range(0, len(self.text), chunk_size)
        ]

    # Sliding window
    def sliding_window_chunking(self, window_size=50, overlap=10):
        chunks = []
        start = 0
        while start < len(self.text):
            end = min(start + window_size, len(self.text))
            chunk = self.text[start:end]
            chunks.append(self._make_metadata(chunk, start, end))
            start += window_size - overlap
        return chunks

    # Semantic chunking
    def semantic_chunking(self):
        if not self.nlp:
            raise ValueError("spaCy model not loaded. Run: python -m spacy download en_core_web_sm")
        doc = self.nlp(self.text)
        chunks, idx = [], 0
        for sent in doc.sents:
            chunk = sent.text.strip()
            start_idx = self.text.find(chunk, idx)
            end_idx = start_idx + len(chunk)
            idx = end_idx
            chunks.append(self._make_metadata(chunk, start_idx, end_idx))
        return chunks

    # Recursive chunking
    def recursive_chunking(self, max_chars=60):
        paragraphs = self.paragraph_chunking()
        chunks = []
        for p in paragraphs:
            para = p["chunk_text"]
            if len(para) <= max_chars:
                chunks.append(p)
            else:
                for i in range(0, len(para), max_chars):
                    chunk = para[i:i+max_chars]
                    start_idx = self.text.find(chunk)
                    end_idx = start_idx + len(chunk)
                    chunks.append(self._make_metadata(chunk, start_idx, end_idx))
        return chunks

    # Token-based
    def token_chunking(self, chunk_size=50, model="gpt-3.5-turbo"):
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(self.text)
        chunks, idx = [], 0
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk_text = enc.decode(chunk_tokens)
            start_idx = self.text.find(chunk_text, idx)
            end_idx = start_idx + len(chunk_text)
            idx = end_idx
            chunks.append(self._make_metadata(chunk_text, start_idx, end_idx, len(chunk_tokens)))
        return chunks

    # Smart Auto Chunking
    def auto_chunking(self):
        length = len(self.text)
        if length < 500:
            print("Using Sentence-level chunking")
            return self.sentence_chunking()
        elif length < 2000:
            print("Using Paragraph-level chunking")
            return self.paragraph_chunking()
        elif length < 5000:
            print("Using Fixed-size chunking")
            return self.fixed_size_chunking(200)
        else:
            print("Using Token-based chunking")
            return self.token_chunking(200)


# ------------------------------
# 🔹 Example usage with dynamic path
# ------------------------------

# Option 1: Direct text
sample_text = "Artificial Intelligence is shaping the future."
chunker1 = TextChunker(text=sample_text)

# Option 2: Load text from file (dynamic path)
file_path = r"F:\My_Afterwork_Projects\Technology\MyProjects\Python\GenAi\Chunks\TempFiles\InputFile\SampleInput_1.txt"
chunker2 = TextChunker(file_path=file_path)

# Run auto chunking
auto_chunks = chunker2.auto_chunking()
for c in auto_chunks:
    print(c)
