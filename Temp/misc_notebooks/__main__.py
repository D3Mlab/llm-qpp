from embedders import RandomEmbedder

if __name__ == "__main__":
    embedder = RandomEmbedder()
    text = "Hello, world!"
    embedding = embedder.embed(text)
    print(f"Embedding for '{text}': {embedding}")
