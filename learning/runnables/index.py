from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableSequence
from dotenv import load_dotenv
load_dotenv()
import os

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
model = ChatNVIDIA(
    #   model="z-ai/glm-5.2",
    model="nvidia/nemotron-3-ultra-550b-a55b",
    #  model="nvidia/nemotron-3-super-120b-a12b",
    api_key=NVIDIA_API_KEY , 
    temperature=1,
    top_p=1,
    max_completion_tokens=16384,
    seed=42,
  
)

prompt1=PromptTemplate(
    template="genrate a summery on this {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="genrate a 5 lines for linkedin post on {topic}",
    input_variables=['topic']
)


parser=StrOutputParser()

parallel_chain=RunnableParallel({
    "toipc_genrate":RunnableSequence(prompt1,model,parser),
    "linkedin":RunnableSequence(prompt2,model,parser)    
})
result=parallel_chain.invoke({"topic":"ai/ml"})
print("result 1",result['toipc_genrate'])
print("result 2",result['linkedin'])
