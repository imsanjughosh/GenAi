import nltk
import spacy
import tiktoken  # For token-level chunking
import os

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
# 🔹 Dynamic file reading
# ------------------------------
def read_file():
    """Ask for a file path dynamically (fallback to default)."""
    default_path = "F:\\My_Afterwork_Projects\\Technology\\MyProjects\\Python\\GenAi\\Chunks\\TempFiles\\InputFile\\SampleInput_1.txt"
    file_path = input(f"Enter full path of your .txt file (Press Enter to use default): ").strip()

    if not file_path:  # if user presses enter, use default
        file_path = default_path

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"✅ File loaded: {file_path}\n")
    return text


# ------------------------------
# 🔹 Example usage
# ------------------------------
if __name__ == "__main__":
    text = read_file()
    if text:
        chunker = TextChunker(text)
        print("TOKEN CHUNKS:", chunker.token_chunking(20))
