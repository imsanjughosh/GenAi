def Print_After_chunking(chunks):
    print("------------------Start Chunking-------------------------")
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1}:", c)
    print("------------------End Chunking-------------------------")