from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

documet = [
    'Delhi is the capital of India',
    'Paris is the capital of France',
    'Lucknow is the capital of Uttar Pradesh'
]

query_embedding = embeddings.embed_documents(documet)

print('Query Embedding Lenght:', len(query_embedding[0]))
print(query_embedding[0][:5])