#GOAL OF THIS FILE

#| Responsibility                | Why                |
#| ----------------------------- | ------------------ |
#| Convert chunks into vectors   | semantic meaning   |
#| Create embeddings             | semantic retrieval |
#| Prepare vectors for vector DB | similarity search  |


from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_embedding(text):

    embedding = embedding_model.embed_query(text)

    return embedding

if __name__ == "__main__":
        sample_text = "Artificial Intelligence improves education"

        embedding = create_embedding(sample_text)

        print("VECTOR LENGTH:")
        print(len(embedding))

        print("\nFIRST 10 VALUES:")
        print(embedding[:10])