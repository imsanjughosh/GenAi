import os
import nltk
import spacy

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

        # Load spaCy model
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


# ------------------------------
# 🔹 Example usage:
# ------------------------------
sample_text = """
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
It has become an essential part of the technology industry. AI research is highly technical and specialized.
The core problems of AI include programming computers for certain traits such as knowledge, reasoning, problem-solving, perception, learning, and planning.
"""

# Static
chunker = TextChunker(text=sample_text)

# Dynamic
file_path = r"F:\My_Afterwork_Projects\Technology\MyProjects\Python\GenAi\Chunks\TempFiles\InputFile\SampleInput_1.txt"
#chunker = TextChunker(file_path=file_path)

print("\n")
print("\nWORD CHUNKS:", chunker.word_chunking(5))
print("\n")
print("\nSENTENCE CHUNKS:", chunker.sentence_chunking())
print("\n")
print("\nPARAGRAPH CHUNKS:", chunker.paragraph_chunking())
print("\n")
print("\nFIXED SIZE CHUNKS:", chunker.fixed_size_chunking(50))
print("\n")
print("\nSLIDING WINDOW CHUNKS:", chunker.sliding_window_chunking(50, 10))
print("\n")
print("\nSEMANTIC CHUNKS:", chunker.semantic_chunking())
print("\n")
print("\nRECURSIVE CHUNKS:", chunker.recursive_chunking(60))
