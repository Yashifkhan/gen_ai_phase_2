from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal,Annotated
from langchain_core.messages import HumanMessage, SystemMessage,BaseMessage,AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import Annotated
from langgraph.types import interrupt,Command
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=1,
    max_tokens=1024,
)

class chatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    
def chat_node(state:chatState):
    decision=interrupt({
        "type":"approval",
        "reason":"Model is about to anser a user qustion",
        "question":state['messages'][-1].content,
        "instruction":"Apprive this queston yes/no"
    })
    
    if decision["approved"] == "no":
        return {"messages":[AIMessage(content="not approved.")]}
    
    else:
        response=model.invoke(state['messages'])
        return {"messages":[response]}
    
agent=StateGraph(chatState)

agent.add_node("chat",chat_node)
agent.add_edge(START,"chat")
agent.add_edge("chat",END)

checkpointer=MemorySaver()
app=agent.compile(checkpointer=checkpointer)

config={"configurable":{"thread_id":"1234"}}
initial_input = {
    "messages": [
        {
            "role": "user",
            "content": "explain me the normal distribution in data preprocessing"
        }
    ]
}

result=app.invoke(initial_input,config=config)
print("agent output ->>",result)



message=result['__interrupt__'][0].value

print("message",message)
user_input=input(f"model messages - {message} \n approve this question yes/no : ")

final_result=app.invoke(Command(resume={"approved":user_input}),config=config)
print("final output",final_result)