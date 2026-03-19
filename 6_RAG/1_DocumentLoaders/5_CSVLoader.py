from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='students.csv')

docs = loader.load()

print(len(docs))
print(docs[18].page_content)