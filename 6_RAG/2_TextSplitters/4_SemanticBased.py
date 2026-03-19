from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def semantic_merge_with_threshold(docs, threshold=0.75):

    embeddings = OpenAIEmbeddings()

    texts = [doc.page_content for doc in docs]
    vectors = embeddings.embed_documents(texts)

    merged_chunks = []
    current_chunk = texts[0]

    for i in range(1, len(texts)):

        sim = cosine_similarity(
            [vectors[i - 1]],
            [vectors[i]]
        )[0][0]

        if sim >= threshold:
            # Merge with current chunk
            current_chunk += "\n" + texts[i]
        else:
            # Save previous chunk and start new one
            merged_chunks.append(Document(page_content=current_chunk))
            current_chunk = texts[i]

    # Append final chunk
    merged_chunks.append(Document(page_content=current_chunk))

    return merged_chunks


semantic_chunker = RunnableLambda(
    lambda docs: semantic_merge_with_threshold(docs, threshold=0.75)
)

# -----------------------------
# Example usage
# -----------------------------

text = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass.

The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness.
"""

doc = Document(page_content=text)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0
)

small_docs = splitter.split_documents([doc])

semantic_docs = semantic_chunker.invoke(small_docs)

print("Number of semantic chunks:", len(semantic_docs))

for i, chunk in enumerate(semantic_docs):
    print(f"\n--- Semantic Chunk {i} ---\n")
    print(chunk.page_content)