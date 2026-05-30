# =========================
# GOAL OF THIS FILE
# =========================

# Evaluate FINAL ANSWER quality
# Measure answer completeness
# Measure keyword recall
# Industrial RAG answer testing


from engine.chatbot_engine import ask_question


# =========================
# TEST CASES
# =========================

test_cases = [

    {
        "question": "What projects has Mohith completed?",
        "expected_keywords": [

            "G-TRI-BAL",
            "Solo Traveller",
            "Sleep Quality Advisor",
            "Tollywood"
        ]
    },

    {
        "question": "What certifications Mohith mentioned?",
        "expected_keywords": [

            "Data",
            "Analytics",
            "Generative",
            "AI"
        ]
    },

    {
        "question": "What technologies are used?",
        "expected_keywords": [

            "Python",
            "FastAPI",
            "LangChain"
        ]
    },

    {
        "question": "What is the abstract of GVP-MAAA?",
        "expected_keywords": [

            "multi-agent",
            "academic",
            "dashboard"
        ]
    }

]


# =========================
# KEYWORD RECALL
# =========================

def keyword_recall(

    answer,

    expected_keywords
):

    found = 0

    answer_lower = answer.lower()

    for keyword in expected_keywords:

        if keyword.lower() in answer_lower:

            found += 1

    return (

        found,

        len(expected_keywords),

        found / len(expected_keywords)
    )


# =========================
# EVALUATION
# =========================

total_score = 0

print("\n")
print("=" * 80)
print("ANSWER EVALUATION")
print("=" * 80)

for test in test_cases:

    question = test["question"]

    expected_keywords = test["expected_keywords"]

    result = ask_question(question)

    answer = result["answer"]

    found, total, score = keyword_recall(

        answer,

        expected_keywords
    )

    total_score += score

    print("\n")
    print("=" * 80)

    print("QUESTION:")
    print(question)

    print("\nEXPECTED KEYWORDS:")
    print(expected_keywords)

    print("\nANSWER:")
    print(answer)

    print("\nFOUND:")
    print(f"{found}/{total}")

    print("\nRECALL:")
    print(f"{score:.2f}")

    print("=" * 80)


# =========================
# FINAL SCORE
# =========================

average_score = (

    total_score

    /

    len(test_cases)
)

print("\n")
print("=" * 80)

print("FINAL ANSWER RECALL SCORE")

print(f"{average_score:.2f}")

print("=" * 80)