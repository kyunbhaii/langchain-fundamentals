from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text1 = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return f"Name: {self.name}, Age: {self.age}, Grade: {self.grade}"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)

print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")
"""

text2 = """
# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a framework that combines information retrieval with large language models. It enhances factual accuracy by grounding responses in external data sources.

---

## 1. Why RAG is Important

Large language models:
- Hallucinate facts
- Have limited knowledge cutoffs
- Cannot access private data by default

RAG solves these problems by retrieving relevant documents before generating a response.

---

## 2. Core Components of RAG

### 2.1 Document Loading

Documents can come from multiple sources:

- PDFs
- CSV files
- Websites
- Databases
- APIs

Example using LangChain:

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("file.txt")
docs = loader.load()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.MARKDOWN,
    chunk_size = 300,
    chunk_overlap =0
)

chunks = splitter.split_text(text2)

print(len(chunks))
print(f"\n{chunks[2]}")