# debug_application.py

from services.indexing_service import all_chunks

for chunk in all_chunks:

    text = chunk["content"].lower()

    if (
        "application" in text
        or "application no" in text
        or "application number" in text
    ):

        print("\nFOUND")
        print("=" * 80)

        print(chunk["metadata"])

        print(chunk["content"][:1500])