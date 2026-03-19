from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct',
    task='text-generation'
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text: \n {topic}",
    input_variables=['topic']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n {notes}, {quiz}',
    input_variables=['notes', 'quiz'] 
)

parser = StrOutputParser()

paralle_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser,
})

merge_chian = prompt3 | model1 | parser

chain = paralle_chain | merge_chian

result = chain.invoke({
    'topic': 'Worm Holes',
})

print(result)

# chain.get_graph().print_ascii()