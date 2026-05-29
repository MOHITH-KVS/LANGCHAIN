cd "QA- CHATBOT"
.\venv\Scripts\activate

uvicorn api.main:app --reload



1. What certifications are mentioned?
2. What internships are mentioned?
3. What is GVP-MAAA?
4. What are the objectives?
5. What agents are used?



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
