from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

# Prompt
template = PromptTemplate(
    template='Give me only name, age and city of a fictional person \n {format_instructions}',
    input_variables=[],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)
print(type(result))