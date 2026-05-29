import pickle

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("TOTAL CHUNKS:", len(chunks))

found = False

for chunk in chunks:

    content = chunk.get("content", "").lower()

    if (
        "cisco" in content
        or "generative ai" in content
        or "machine learning for finance" in content
        or "data analytics essentials" in content
    ):

        found = True

        print("\n" + "=" * 80)

        print("CHUNK ID:")
        print(chunk.get("chunk_id"))

        print("\nMETADATA:")
        print(chunk.get("metadata"))

        print("\nCONTENT:")
        print(chunk.get("content"))

        print("\n" + "=" * 80)

if not found:
    print("\nNO CERTIFICATION CHUNK FOUND")