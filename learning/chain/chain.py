# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv

# load_dotenv()

# import os
# NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# prompt=PromptTemplate(
#     template="Genrate the 5 lines on {topic}",
#     input_variables=['topic']
# )

# model= ChatNVIDIA(
#     # model="nvidia/nemotron-3-ultra-550b-a55b",
#     model="meta/llama-3.1-8b-instruct",
#     api_key=NVIDIA_API_KEY , 
#     temperature=.5,
#     top_p=1,
#     max_completion_tokens=200,
#     seed=42,
# )
# parser=StrOutputParser()

# chain=prompt | model | parser

# result=chain.invoke({'topic':"ai/ml"})
# print(result)


# advance chain 

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

import os
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

prompt1=PromptTemplate(
    template="Genrate the report on {topic}",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="Genrate the summery on flowing text {text}",
    input_variables=['text']
)

model= ChatNVIDIA(
    # model="nvidia/nemotron-3-ultra-550b-a55b",
    model="meta/llama-3.1-8b-instruct",
    api_key=NVIDIA_API_KEY , 
    temperature=.5,
    top_p=1,
    max_completion_tokens=200,
    seed=42,
)
parser=StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
result=chain.invoke({"topic":"java script"})
print(result)
