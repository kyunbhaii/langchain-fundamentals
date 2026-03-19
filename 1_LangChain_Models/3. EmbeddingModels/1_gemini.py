from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', output_dimensionality=32)

documet = [
    'Delhi is the capital of India',
    'Paris is the capital of France',
    'Lucknow is the capital of Uttar Pradesh'
]

quer_embedding = embedding.embed_documents(documet)

print(str(quer_embedding ))