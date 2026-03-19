from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment : Literal['Positive', 'Negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text.\n\n{format_instructions}\n\nFeedback:\n{feedback}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'Negative', prompt3 | model | parser),
    RunnableLambda(lambda x: 'Could not determine sentiment')
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':'This is phone is not to the hype'})

print(result)

chain.get_graph().print_ascii()