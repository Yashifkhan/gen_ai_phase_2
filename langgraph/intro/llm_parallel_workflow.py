from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph ,START ,END
from typing import TypedDict,Annotated
from pydantic import BaseModel, Field 
from dotenv import load_dotenv
import operator
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

# define the state 
class EvalauteShema(BaseModel):
    feedback:str = Field(description="detailed feed on this essay")
    score:int =Field(description="score out of 10",ge=0,le=10)
    
structure_model=model.with_structured_output(EvalauteShema)

essay="""Artificial Intelligence (AI) is one of the most important technologies shaping the modern world. It enables computers and machines to perform tasks that traditionally require human intelligence, such as understanding language, recognizing images, making predictions, solving problems, and making decisions.

Two important areas of AI are Machine Learning (ML) and Generative AI (Gen AI). Machine Learning allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every situation. Generative AI goes a step further by learning patterns from large amounts of data and using those patterns to create new content such as text, images, audio, video, and code.

Although Machine Learning and Generative AI are closely related, they are not the same. Understanding their relationship is important for anyone who wants to build modern AI applications.

What is Machine Learning?

Machine Learning is a branch of Artificial Intelligence in which computers learn from data.

Traditional programming generally follows:

Data + Rules → Output

For example, if we want to create a program that determines whether a student passes an exam, we might manually write rules such as:

If marks are greater than 40, pass.
Otherwise, fail.

Machine Learning uses a different approach:

Data + Expected Output → Model

The model learns the relationship between the input and output from examples.

For instance, we can provide a Machine Learning model with thousands of examples containing:

Student study hours
Attendance
Previous marks
Assignment scores
Final examination marks

The model can learn patterns from this data and predict the marks of a new student."""
prompt=f"Evaluate the language of flowing essay and give me feed back and assign a score out of 10 \n {essay}"

# result=structure_model.invoke(prompt)

# print("structure model output: --->>", result)
# print("Feedback:", result.feedback)
# print("Score:", result.score)

class UPSCState(TypedDict):
    essay:str
    language_feedback:str
    analysis_feedback:str
    clearity_feedback:str
    indivadual_score: Annotated[list[int],operator.add]
    avg_score:float
    
graph=StateGraph(UPSCState)

def evaluate_lang(state:UPSCState):
    prompt=f"Evaluate the language of flowing essay and give me feed back and assign a score out of 10 \n {state['essay']}"
    output=structure_model.invoke(prompt)
    return {"language_feedback":output.feedback,"indivadual_score":[output.score]}

def evaluate_analysis(state:UPSCState):
    prompt=f"Evaluate the depth of analysis and give me feed back and assign a score out of 10 \n {state['essay']}"
    output=structure_model.invoke(prompt)
    return {"analysis_feedback":output.feedback,"indivadual_score":[output.score]}

def evaluate_thought(state:UPSCState):
    prompt=f"Evaluate the clarity of thought of flowing essay and give me feed back and assign a score out of 10 \n {state['essay']}"
    output=structure_model.invoke(prompt)
    return {"clearity_feedback":output.feedback,"indivadual_score":[output.score]}

def final_evaluate(state:UPSCState):
    prompt=f"based the flowing feed back create a summarized feedback \n language feedback - {state['language_feedback']} \n depth of  analysis feedback - {state['analysis_feedback']} \n clearity of thought feedback {state['clearity_feedback']}"
    overall_feedback=model.invoke(prompt)
    avg_score=sum(state['indivadual_score'])/len(state['indivadual_score'])
    return {'overall_language':overall_feedback,'avg_score':avg_score}



# nodes 
graph.add_node('evaluate_lang',evaluate_lang)
graph.add_node('evaluate_analysis',evaluate_analysis)
graph.add_node('evaluate_thought',evaluate_thought)
graph.add_node('final_evaluate',final_evaluate)

# edgs 
graph.add_edge(START,'evaluate_lang')
graph.add_edge(START,'evaluate_analysis')
graph.add_edge(START,'evaluate_thought')


graph.add_edge('evaluate_lang','final_evaluate')
graph.add_edge('evaluate_analysis','final_evaluate')
graph.add_edge('evaluate_thought','final_evaluate')

graph.add_edge('final_evaluate',END)

workflow=graph.compile()

result=workflow.invoke({"essay":essay})
print("result is : ", result)
