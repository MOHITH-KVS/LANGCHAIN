# ============================================
# GOAL OF THIS FILE
# ============================================

# This file should ONLY:
#
# 1. Trigger indexing pipeline
# 2. Run document processing
# 3. Build vector indexes
# 4. Build BM25 indexes
# 5. Print indexing diagnostics
# 6. Help debug chunk quality


# ============================================
# IMPORTS
# ============================================

from services.indexing_service import index_documents

import traceback


# ============================================
# MAIN ENTRYPOINT
# ============================================

def main():

    print("\n" + "=" * 80)
    print("STARTING INDEXING PIPELINE")
    print("=" * 80)

    try:

        # ============================================
        # RUN INDEXING
        # ============================================

        chunks = index_documents()


        # ============================================
        # VALIDATION
        # ============================================

        if not chunks:

            print("\n[ERROR] No chunks generated.")
            return


        print("\n" + "=" * 80)
        print("INDEXING COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print(f"\nTOTAL CHUNKS GENERATED: {len(chunks)}")


        # ============================================
        # DEBUG CHUNK OUTPUT
        # ============================================

        print("\n" + "=" * 80)
        print("ALL GENERATED CHUNKS")
        print("=" * 80)


        for i, chunk in enumerate(chunks):

            print(f"\nCHUNK #{i + 1}")

            print("-" * 80)

            metadata = chunk.get("metadata", {})

            print(f"SOURCE  : {metadata.get('source', 'Unknown')}")
            print(f"PAGE    : {metadata.get('page', 'Unknown')}")
            print(f"SECTION : {metadata.get('section', 'general')}")

            print("\nCONTENT:\n")

            print(chunk.get("content", ""))

            print("\n" + "=" * 80)


        print("\nINDEXING DEBUG COMPLETED")


    except Exception as e:

        print("\n" + "=" * 80)
        print("INDEXING FAILED")
        print("=" * 80)

        print(f"\nERROR: {str(e)}\n")

        traceback.print_exc()


# ============================================
# SCRIPT ENTRY
# ============================================

if __name__ == "__main__":

    main()