from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path='PDFs/NLP.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=21,
    separator=''
)

chunks = splitter.split_documents(docs)

print(chunks[21].page_content)
print(f'\nNumber of Chunks: {len(chunks)}')