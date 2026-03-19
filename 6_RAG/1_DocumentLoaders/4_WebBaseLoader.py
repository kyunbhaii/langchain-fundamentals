from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

prompt = PromptTemplate(
    template='Answer the following quesntion \n{question} from the following text: \n{text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url = 'https://www.bbc.com/news/articles/c86yjnw4x49o'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'question':'What is the main point of this article?','text':docs[0].page_content})

# print(len(docs))
# print(docs[0].page_content)
print(result)