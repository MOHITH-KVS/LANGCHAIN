# GOAL OF THIS FILE

# This file should ONLY:

#| Responsibility              | Why                              |
#| --------------------------- | -------------------------------- |
#| Run indexing pipeline       | execution entrypoint             |
#| Trigger document indexing   | reusable service architecture    |


from services.indexing_service import (

    index_documents
)


# =========================
# RUN INDEXING
# =========================

if __name__ == "__main__":

    index_documents()