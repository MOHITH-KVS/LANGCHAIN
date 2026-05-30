import pickle

# =========================
# LOAD CHUNKS
# =========================

with open("chunks.pkl", "rb") as f:

    chunks = pickle.load(f)

print("\nTOTAL CHUNKS:", len(chunks))


# =========================
# PDF TO INSPECT
# =========================

target_pdf = "4954295325.pdf"


# =========================
# KEYWORDS TO SEARCH
# =========================

keywords = [

    "application",
    "application no",
    "application number",
    "4954295325",
    "driving",
    "test",
    "address",
    "rta",
    "visakhapatnam"
]


# =========================
# SEARCH KEYWORDS
# =========================

print("\n")
print("=" * 100)
print("KEYWORD SEARCH RESULTS")
print("=" * 100)

found_count = 0

for chunk in chunks:

    text = chunk["content"].lower()

    for keyword in keywords:

        if keyword in text:

            found_count += 1

            print("\n")
            print("=" * 100)

            print(f"FOUND KEYWORD: {keyword}")

            print("\nMETADATA:")

            print(chunk["metadata"])

            print("\nCONTENT:")

            print(chunk["content"][:1500])

            print("=" * 100)

            break


# =========================
# INSPECT SPECIFIC PDF
# =========================

print("\n")
print("=" * 100)
print(f"ALL CHUNKS FROM: {target_pdf}")
print("=" * 100)

pdf_chunk_count = 0

for chunk in chunks:

    source = chunk["metadata"].get("source", "")

    if target_pdf.lower() in source.lower():

        pdf_chunk_count += 1

        print("\n")
        print("=" * 100)

        print("METADATA:")

        print(chunk["metadata"])

        print("\nCONTENT:")

        print(chunk["content"][:3000])

        print("=" * 100)


# =========================
# SUMMARY
# =========================

print("\n")
print("=" * 100)

print("TOTAL KEYWORD MATCHES:", found_count)

print("TOTAL CHUNKS FROM PDF:", pdf_chunk_count)

print("=" * 100)