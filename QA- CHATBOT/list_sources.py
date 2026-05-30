import pickle

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

sources = {}

for chunk in chunks:

    source = chunk["metadata"].get("source", "UNKNOWN")

    sources[source] = sources.get(source, 0) + 1

print("\nDOCUMENTS IN CHUNKS DATABASE\n")

for source, count in sorted(sources.items()):

    print(f"{source} --> {count} chunks")