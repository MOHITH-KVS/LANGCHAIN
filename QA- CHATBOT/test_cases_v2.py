# =========================
# GOAL OF THIS FILE
# =========================

# Automated End-to-End RAG Evaluation
# Tests retrieval + reranking + generation
# Measures answer quality
# Reports PASS / PARTIAL / FAIL


from engine.chatbot_engine import ask_question


# =========================
# TEST CASES
# =========================

test_cases = [

    {
        "question": "What certifications does Mohith have?",
        "expected_keywords": [
            "analytics",
            "generative ai",
            "machine learning"
        ]
    },

    {
        "question": "What skills does Mohith have?",
        "expected_keywords": [
            "python",
            "sql",
            "power bi"
        ]
    },

    {
        "question": "What projects has Mohith completed?",
        "expected_keywords": [
            "sleep",
            "advisor"
        ]
    },

    {
        "question": "What is Mohith's CGPA?",
        "expected_keywords": [
            "8.9"
        ]
    },

    {
        "question": "Explain abstract",
        "expected_keywords": [
            "project"
        ]
    },

    {
        "question": "Explain methodology",
        "expected_keywords": [
            "method"
        ]
    },

    {
        "question": "Explain architecture",
        "expected_keywords": [
            "architecture"
        ]
    },

    {
        "question": "Explain conclusion",
        "expected_keywords": [
            "conclusion"
        ]
    },

    {
        "question": "What technologies are used?",
        "expected_keywords": [
            "technology"
        ]
    },

    {
        "question": "What problem does the project solve?",
        "expected_keywords": [
            "problem"
        ]
    },

    {
        "question": "Explain implementation",
        "expected_keywords": [
            "implementation"
        ]
    },

    {
        "question": "What are the objectives?",
        "expected_keywords": [
            "objective"
        ]
    },

    {
        "question": "Summarize the project",
        "expected_keywords": [
            "project"
        ]
    },

    {
        "question": "What datasets are used?",
        "expected_keywords": [
            "data"
        ]
    },

    {
        "question": "What results were achieved?",
        "expected_keywords": [
            "result"
        ]
    },

    # Negative tests

    {
        "question": "Who won IPL 2025?",
        "expected_keywords": []
    },

    {
        "question": "What is the capital of Brazil?",
        "expected_keywords": []
    },

    {
        "question": "Who is Elon Musk?",
        "expected_keywords": []
    },

    {
        "question": "Explain quantum teleportation",
        "expected_keywords": []
    },

    {
        "question": "Teach me Python programming",
        "expected_keywords": []
    }

]


# =========================
# EVALUATION
# =========================

pass_count = 0
partial_count = 0
fail_count = 0

print("\n")
print("=" * 80)
print("RAG END-TO-END EVALUATION")
print("=" * 80)

for idx, test in enumerate(test_cases, start=1):

    question = test["question"]
    expected_keywords = test["expected_keywords"]

    print("\n")
    print("=" * 80)
    print(f"TEST {idx}")
    print("=" * 80)

    try:

        response = ask_question(question)

        answer = response["answer"].lower()

        if len(expected_keywords) == 0:

            # Negative test

            if (
                "not available" in answer
                or "not found" in answer
                or "context" in answer
            ):

                result = "PASS"
                pass_count += 1

            else:

                result = "FAIL"
                fail_count += 1

        else:

            matches = 0

            for keyword in expected_keywords:

                if keyword.lower() in answer:

                    matches += 1

            ratio = matches / len(expected_keywords)

            if ratio >= 0.75:

                result = "PASS"
                pass_count += 1

            elif ratio >= 0.40:

                result = "PARTIAL"
                partial_count += 1

            else:

                result = "FAIL"
                fail_count += 1

        print("QUESTION:")
        print(question)

        print("\nRESULT:")
        print(result)

    except Exception as e:

        print("\nERROR:")
        print(str(e))

        result = "FAIL"
        fail_count += 1


# =========================
# FINAL REPORT
# =========================

total_tests = len(test_cases)

accuracy = (

    (pass_count + (0.5 * partial_count))

    / total_tests

) * 100

print("\n")
print("=" * 80)
print("FINAL REPORT")
print("=" * 80)

print(f"TOTAL TESTS : {total_tests}")
print(f"PASS        : {pass_count}")
print(f"PARTIAL     : {partial_count}")
print(f"FAIL        : {fail_count}")

print("\nACCURACY:")
print(f"{accuracy:.2f}%")

print("=" * 80)