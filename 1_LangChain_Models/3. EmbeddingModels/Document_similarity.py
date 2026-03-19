from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model='gemini-embedding-001',
    output_dimensionality=300
)

documents = [
    "New Delhi is the capital city of India.",
    "Paris is the capital of France and a major tourist destination.",
    "Lucknow is the capital of the Indian state Uttar Pradesh.",
    "India is a country in South Asia with a large population.",
    "France is a European country known for art, food, and culture.",
    "Uttar Pradesh is one of the most populous states in India.",
    "Berlin is the capital of Germany.",
    "Germany is a country located in Central Europe."
]

query = 'France is known for?'

doc_embedding = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

# print('Each Embedding Lenght:', len(doc_embedding[0]))
# print(scores)
print(f'Question: {query}')
print(f'Answer: {documents[index]}')
print(f'The similarity score is {score}')