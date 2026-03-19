from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template='Write about this {topic} for linkeding purpose',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Write about this {topic} for twitter purpose',
    input_variables=['topic']
)

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(f"Tweet Structure:\n{result['tweet']}\n\n\nLinkedin Post Structure:\n{result['linkedin']}")