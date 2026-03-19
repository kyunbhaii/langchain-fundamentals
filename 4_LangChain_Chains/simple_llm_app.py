from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Suggest a catchy blog title about {topic}.",
    input_variables=['topic']
)

topic = input('Enter a Topic: ')

chain = prompt | model | parser

result = chain.invoke({"topic": topic})

print(f'Generated Blog Title: {result}')