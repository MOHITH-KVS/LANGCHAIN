import pickle

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("\nINSPECTING 4954295325.pdf\n")

for i, chunk in enumerate(chunks):

    source = chunk["metadata"].get("source", "")

    if source == "4954295325.pdf":

        print("\n" + "=" * 100)
        print(f"CHUNK {i}")
        print("=" * 100)

        print("\nMETADATA:")
        print(chunk["metadata"])

        print("\nCONTENT:")
        print(chunk["content"])

        print("\n")