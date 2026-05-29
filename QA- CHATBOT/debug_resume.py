import pickle

with open("chunks.pkl","rb") as f:
    chunks = pickle.load(f)

for chunk in chunks:

    if chunk["chunk_id"] in [569,570,571]:

        print("\n")
        print("="*80)
        print("ID:", chunk["chunk_id"])
        print("SOURCE:", chunk["metadata"]["source"])
        print("SECTION:", chunk["metadata"]["section"])
        print(chunk["content"])