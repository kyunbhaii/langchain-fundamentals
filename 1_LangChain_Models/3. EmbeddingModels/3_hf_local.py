from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

document = [
    'Delhi is the capital of India',
    'Paris is the capital of France',
    'Lucknow is the capital of Uttar Pradesh'
]

doc_embedding = embedding.embed_documents(document)

print('Lenght of the Embeddings',len(doc_embedding[0]))
print(str(doc_embedding[0][:5]))