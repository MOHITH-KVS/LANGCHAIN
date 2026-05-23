cd "QA- CHATBOT"
.\venv\Scripts\activate

uvicorn api.main:app --reload

| File Changed           | Re-run What?         |
| ---------------------- | -------------------- |
| generation.py          | chatbot.py           |
| reranker.py            | chatbot.py           |
| query_rewriter.py      | chatbot.py           |
| chatbot.py             | chatbot.py           |
| chunking.py            | indexing_pipeline.py |
| embeddings.py          | indexing_pipeline.py |
| document_processor.py  | indexing_pipeline.py |
| vector_store structure | indexing_pipeline.py |
| new PDFs               | indexing_pipeline.py |
