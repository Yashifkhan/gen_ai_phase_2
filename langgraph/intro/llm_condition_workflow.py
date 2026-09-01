from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph ,START ,END
from typing import TypedDict,Annotated,Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import operator
load_dotenv()
import os

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
model = ChatNVIDIA(
    # model="nvidia/nemotron-3-super-120b-a12b",
    # model="meta/llama-3.1-8b-instruct",
    model="nvidia/nemotron-3-ultra-550b-a55b",
    api_key=NVIDIA_API_KEY , 
    temperature=1,
    top_p=1,
    max_completion_tokens=1024,
    seed=42,
)

class SentimentSchema(BaseModel):
    sentiment:Literal['positive','negative'] =Field(description="sentiment of the review")

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(
        description="The category of the issue"
    )
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(
        description="The emotional tone of the user"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent or critical the issue is"
    )

structure_model=model.with_structured_output(SentimentSchema)
structure_model2=model.with_structured_output(DiagnosisSchema)

# define the state 
class SentimetnState(TypedDict):
    sentiment:Literal['positive','negative']
    review:str
    response:str
    diagnosis:dict

# create graph 
agent=StateGraph(SentimetnState)

def gen_sentiment(state:SentimetnState):
    prompt=f"What is the sentiment of the flowing review-the {state['review']}"
    result=structure_model.invoke(prompt).sentiment 
    state['sentiment']=result
    return {'sentiment':result}

def check_condition(state:SentimetnState) -> Literal['run_diagnosis','positive_reponse']:
    
    if state['sentiment'] == 'positive':
        return 'positive_reponse'
    else:
        return 'run_diagnosis'
       
def run_diagnosis(state:SentimetnState):
   prompt=f"giagnosis the negitive review :: \n\n {state['review']} \n\n return issue type tone and urgency"
   response=structure_model2.invoke(prompt)
   state['response']=response
   return {"diangnosis":response.model_dump}

def positive_reponse(state:SentimetnState):
    prompt=f"write a warm than-you in response to this review : \n\n {state['review']} \n\n also kind the user to leave feedback on our website."
    response=model.invoke(prompt)
    state['response']=response
    return {'response':response}

def negitive_response(state:SentimetnState):
    diagnosis=state["diagnosis"]
    prompt=f"""You are a support assistant.
    The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as
    '{diagnosis['urgency']}'.
    Write an empathetic, helpful resolution message."""
    response=model.invoke(prompt).content
    state['diagnosis']=response
    return {"response":response}

# create a node 
agent.add_node("genrate_sentiment",gen_sentiment)
agent.add_node('positive_reponse',positive_reponse)
agent.add_node('run_diagnosis',run_diagnosis)
agent.add_node('negitive_response',negitive_response)

# add edge 
agent.add_edge(START,'genrate_sentiment')
agent.add_conditional_edges('genrate_sentiment',check_condition)
# agent.add_edge('genrate_sentiment','negitive_response')
agent.add_edge('run_diagnosis', 'negitive_response')

agent.add_edge('positive_reponse',END)
agent.add_edge('negitive_response',END)


workflow=agent.compile()

result=workflow.invoke({"review":"""I've been using this app for about a month now, and I must say, the user interface is incredibly clean and intuitive.
Everything is exactly where you'd expect it to be. It's rare to find something that just works without needing a tutorial.
Great job to the design team!"""})
print("final output : -->>",result)
