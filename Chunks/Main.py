# main.py
from word_chunking import word_chunking 
from Sliding_Window_Chunking import sliding_window_chunking
from Sentence_level_Cunking import sentence_chunking
from Semantic_Chunking import semantic_chunking
from Recursive_Chunking import recursive_chunking
from Paragraph_level_chunking import paragraph_chunking
from Fixed_size_chunking import fixed_size_chunking
from Print_Chunks import Print_After_chunking

def read_file():
    """Ask for a file path and read its content."""
    #file_path = input("Enter the full path of your .txt file: ").strip()
    file_path = "F:\My_Afterwork_Projects\Technology\MyProjects\Python\GenAi\Chunks\TempFiles\InputFile\SampleInput_1.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print("\nFile loaded successfully!\n")
        return text
    except FileNotFoundError:
        print("File not found. Please try again.")
        return None

def main():
    text = None
    while text is None:
        text = read_file()

    # Mapping menu numbers to functions
    functions = {
        "1": word_chunking,
        "2": sliding_window_chunking,
        "3": sentence_chunking,
        "4": semantic_chunking,
        "5": recursive_chunking,
        "6": paragraph_chunking,
        "7": fixed_size_chunking,
    }

    while True:
        print("\nChoose a Type of Chunking:")
        print("1. Word-level chunking (Fixed number of words per chunk) \n2. Sliding Window Chunking (Overlap between chunks) \n3. Sentence-level chunking \n4. Semantic Chunking (Split based on meaning) \n5. Recursive Chunking (Hierarchical)")
        print("6. Paragraph-level chunking \n7. Fixed-size chunking (Character-based) ")
        print("Type 'exit' to quit.")
        
        choice = input("Enter your choice: ").strip()

        if choice.lower() == "exit":
            print("Exiting program...")
            break
        elif choice in functions:
            chunks = functions[choice](text)
            Print_After_chunking(chunks)
        else:
            print("Invalid choice! Try again.")

if __name__ == "__main__":
    main()
