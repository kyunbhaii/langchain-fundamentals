# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0
# )

# # 1st Prompt -> Detailed Report
# template1 = PromptTemplate(
#     template='Write a detailed report on {topic}',
#     input_variables=['topic']
# )

# # 2nd prompt -> summary
# template2 = PromptTemplate(
#     template='Write down 5 line summary on the following text. /n {text}',
#     input_variables=['text']
# )

# parser = StrOutputParser()

# chain = template1 | model | parser | template2 | model | parser

# result = chain.invoke({'topic':'black hole'})

# print(result)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# 1st Prompt -> Detailed Report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write down 5 line summary on the following text.\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "black hole"})

print(result)