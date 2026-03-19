from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarise the following text:\n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 300,
        RunnableLambda(lambda x: {"text": x}) | prompt2 | model | parser
    ),
    RunnablePassthrough()
)

final_chain = report_gen_chain | branch_chain

result = final_chain.invoke({'topic': 'Attention is All You Need research paper'})

print(result)
print(len(result.split()))