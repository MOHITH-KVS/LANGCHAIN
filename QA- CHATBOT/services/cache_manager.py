# =========================
# RESPONSE CACHE
# =========================

response_cache = {}


# =========================
# GET CACHE
# =========================

def get_cached_response(question):

    return response_cache.get(

        question.lower().strip()
    )


# =========================
# SAVE CACHE
# =========================

def save_to_cache(

    question,

    response
):

    response_cache[

        question.lower().strip()

    ] = response