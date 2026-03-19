# Dont use run use this, it takes a lot of time for inference on local

# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# llm = HuggingFacePipeline.from_model_id(
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task = 'text-generation',
#     pipeline_kwargs = dict(
#         temperature = 0.1,
#         max_new_tokens = 100
#     )
# )

# model = ChatHuggingFace(llm = llm)

# response = model.invoke('What is the capital USA')

# print(response.content)