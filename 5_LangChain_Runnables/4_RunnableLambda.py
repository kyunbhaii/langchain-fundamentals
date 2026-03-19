from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

def word_counter(text):
    return len(text.split())

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

parser = StrOutputParser()

runnable_word_counter = RunnableLambda(word_counter)

joke_gen_chain = prompt | model | parser

parallel_chain = RunnableParallel({
    "joke": joke_gen_chain,
    "total_word": joke_gen_chain | runnable_word_counter,
})

final_chain = parallel_chain

result = final_chain.invoke({'topic':'week'})

final_output = """{}\n\nTotal Words: {}""".format(result['joke'], result['total_word'])

print(final_output)