from langchain_text_splitters import RecursiveCharacterTextSplitter

with open('poem.txt', 'r', encoding='utf-8') as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 250,
    chunk_overlap = 21
)

chunks = splitter.split_text(text)

print(f'Number of Chunks: {len(chunks)}')
print(f'\n{chunks[3]}')