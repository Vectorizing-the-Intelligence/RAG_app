# RAG_app
A simple application for the Retrieval-Augmented Generation (RAG) system.

In this RAG system, the following steps are performed:
1. The user uploads a file (PDF or Word).
2. The file is partitioned into paragraphs, with each paragraph saved as a Parquet file.
3. The Stella model converts each paragraph into an embedding vector.
4. The user asks a question about the uploaded documents.
5. The application embeds the query into a vector, similar to how the document paragraphs were processed.
6. The database searches through all stored vectors and retrieves the top 5 most relevant paragraphs.
7. The user selects a preferred model (T5 or Gemini) for the generation phase.
8. The query, along with the top relevant paragraphs, is passed to the selected model, and the generated answer is displayed to the user.
