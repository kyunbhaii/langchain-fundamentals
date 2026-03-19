from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    cultivation: str = Field(description="Cultivation realm of the person")
    martial_art_technique: str = Field(description="Martial art technique of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="""Provide name, cultivation, and martial art technique of a fictional {type} novel character. {format_instructions}""",
    input_variables=["type"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({"type": "chinese martial arts"})

print(result)
# print(f'Prompt: \n {template.invoke({"type": "chinese martial arts"})}')
# print(type(result))