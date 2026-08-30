from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph ,START ,END
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()
import os

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
print("NVIDIA_API_KEY",NVIDIA_API_KEY)
model = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    api_key=NVIDIA_API_KEY , 
    temperature=1,
    top_p=1,
    max_completion_tokens=16384,
    seed=42,
)

class llmState(TypedDict):
    qu:str
    answer:str
    
graph=StateGraph(llmState)

# define the graph 
def llm_aq(state:llmState) ->llmState:
    question=state["qu"]
    prompt=f'Answer the flowing question {question}'
    
    answer=model.invoke(prompt).content
    state["answer"]=answer
    return state


graph.add_node("llm_qa",llm_aq)
graph.add_edge(START,'llm_qa')
graph.add_edge('llm_qa',END)

workflow=graph.compile()
result=workflow.invoke({"qu":"what is ai/ml ? "})
print("Result : ",result)