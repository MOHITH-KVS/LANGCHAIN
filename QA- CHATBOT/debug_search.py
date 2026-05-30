from core.vector_store import collection

results = collection.get()

print(f"\nTotal Chunks: {len(results['documents'])}")

for i, doc in enumerate(results["documents"]):

    if "application" in doc.lower():

        print("\nFOUND")
        print("-" * 50)
        print(doc[:1000])