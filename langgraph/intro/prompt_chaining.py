from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph ,START ,END
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()
import os

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
model = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    api_key=NVIDIA_API_KEY , 
    temperature=1,
    top_p=1,
    max_completion_tokens=1024,
    seed=42,
)

class BlogState(TypedDict):
    title:str
    outline:str
    content:str

graph=StateGraph(BlogState)

def create_outline(state:BlogState) -> BlogState:
    title=state["title"]
    prompt=f"genrate the outline flowing topic {title}"
    outline=model.invoke(prompt).content
    state["outline"]=outline
    return state

def create_blog(state:BlogState) -> BlogState:
    title=state["title"]
    outline=state["outline"]
    prompt=f"write a blog on the title   is {title} using the outline is \n {outline}"
    content=model.invoke(prompt).content
    state["content"]=content
    return state
    

graph.add_node("create_outline",create_outline)
graph.add_node("create_blog",create_blog)

graph.add_edge(START,'create_outline')
graph.add_edge('create_outline','create_blog')
graph.add_edge('create_blog',END)

workflow=graph.compile()
result=workflow.invoke({'title':'rise of ai in india'})
print(result)