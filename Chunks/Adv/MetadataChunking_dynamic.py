import nltk
import spacy
import tiktoken

# Download NLTK punkt tokenizer
nltk.download('punkt', quiet=True)

class TextChunker:
    def __init__(self, text):
        self.text = text
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please run: python -m spacy download en_core_web_sm")
            self.nlp = None

    # 🔹 Helper: Create metadata dictionary
    def _make_metadata(self, chunk, start_idx, end_idx, token_count=None):
        return {
            "chunk_text": chunk,
            "start_index": start_idx,
            "end_index": end_idx,
            "token_count": token_count
        }

    # 1. Word-level chunking
    def word_chunking(self, chunk_size=5):
        words = self.text.split()
        chunks = []
        idx = 0
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            start_idx = self.text.find(chunk, idx)
            end_idx = start_idx + len(chunk)
            idx = end_idx
            chunks.append(self._make_metadata(chunk, start_idx, end_idx))
        return chunks

    # 2. Sentence-level chunking
    def sentence_chunking(self):
        sentences = nltk.sent_tokenize(self.text)
        chunks = []
        idx = 0
        for sent in sentences:
            start_idx = self.text.find(sent, idx)
            end_idx = start_idx + len(sent)
            idx = end_idx
            chunks.append(self._make_metadata(sent, start_idx, end_idx))
        return chunks

    # 3. Paragraph-level chunking
    def paragraph_chunking(self):
        paragraphs = [p.strip() for p in self.text.split("\n") if p.strip()]
        chunks = []
        idx = 0
        for para in paragraphs:
            start_idx = self.text.find(para, idx)
            end_idx = start_idx + len(para)
            idx = end_idx
            chunks.append(self._make_metadata(para, start_idx, end_idx))
        return chunks

    # 4. Fixed-size chunking (characters)
    def fixed_size_chunking(self, chunk_size=50):
        return [
            self._make_metadata(self.text[i:i+chunk_size], i, i+chunk_size)
            for i in range(0, len(self.text), chunk_size)
        ]

    # 5. Sliding window chunking
    def sliding_window_chunking(self, window_size=50, overlap=10):
        chunks = []
        start = 0
        while start < len(self.text):
            end = start + window_size
            chunk = self.text[start:end]
            chunks.append(self._make_metadata(chunk, start, end))
            start += window_size - overlap
        return chunks

    # 6. Semantic chunking
    def semantic_chunking(self):
        if not self.nlp:
            raise ValueError("spaCy model not loaded. Run: python -m spacy download en_core_web_sm")
        doc = self.nlp(self.text)
        chunks = []
        idx = 0
        for sent in doc.sents:
            chunk = sent.text.strip()
            start_idx = self.text.find(chunk, idx)
            end_idx = start_idx + len(chunk)
            idx = end_idx
            chunks.append(self._make_metadata(chunk, start_idx, end_idx))
        return chunks

    # 7. Recursive chunking
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

    # 8. Token-based chunking
    def token_chunking(self, chunk_size=50, model="gpt-3.5-turbo"):
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(self.text)
        chunks = []
        idx = 0
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk_text = enc.decode(chunk_tokens)
            start_idx = self.text.find(chunk_text, idx)
            end_idx = start_idx + len(chunk_text)
            idx = end_idx
            chunks.append(self._make_metadata(chunk_text, start_idx, end_idx, len(chunk_tokens)))
        return chunks

    # 9. Auto chunking (new method)
    def auto_chunking(self, max_chars=150):
        """
        Automatically choose the best chunking strategy:
        - Short text → one chunk
        - Medium text → sentence-based
        - Long text → recursive
        """
        if len(self.text) <= max_chars:
            return [self._make_metadata(self.text, 0, len(self.text))]
        elif len(self.text) <= max_chars * 3:
            return self.sentence_chunking()
        else:
            return self.recursive_chunking(max_chars=max_chars)


def read_file():
    """Ask for a file path and read its content."""
    file_path = "F:\\My_Afterwork_Projects\\Technology\\MyProjects\\Python\\GenAi\\Chunks\\TempFiles\\InputFile\\SampleInput_1.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print("\nFile loaded successfully!\n")
        return text
    except FileNotFoundError:
        print("File not found. Please try again.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    text = read_file()
    if text:
        chunker = TextChunker(text)
        auto_chunks = chunker.auto_chunking(max_chars=150)

        print("Generated Chunks:\n")
        for i, c in enumerate(auto_chunks, 1):
            print(f"Chunk {i}: {c}\n")
