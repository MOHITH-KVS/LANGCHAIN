# debug_registry.py

import pickle

with open("document_registry.pkl", "rb") as f:
    registry = pickle.load(f)

for doc_name, profile in registry.items():

    print("\n" + "="*80)
    print("DOCUMENT:", doc_name)

    print("\nSUMMARY:")
    print(profile.get("summary",""))

    print("\nKEYWORDS:")
    print(profile.get("keywords",[])[:30])

    print("\nSAMPLE CHUNKS:")
    for chunk in profile.get("sample_chunks",[])[:3]:
        print(chunk[:300])