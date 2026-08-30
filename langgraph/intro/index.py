from langgraph.graph import StateGraph,START ,END
from typing import TypedDict

# define the state 
class BMIState(TypedDict):
    weight_kg:float
    height_m:float
    bmi:float
    category:str

# create graph 
graph=StateGraph(BMIState)

# define the graph 
def BMICalculate(state:BMIState) -> BMIState:
    weaight=state["weight_kg"]
    heaight=state["height_m"]
    bmi=weaight/(heaight ** 2)
    state["bmi"]= round(bmi ,2)
    return state
    
def lableBmi(state:BMIState)-> BMIState:
    bmi=state['bmi']
    if bmi < 18.5:
        state["category"]="underweight"
    elif 18.5 <= bmi <25:
        state["category"]="normal"
    elif 25 <= bmi < 30:    
        state["category"]="overweight"
    else :
        state["category"]="obese"
        
    return state
    
# add node on graph 
graph.add_node("bmi_cal",BMICalculate)
graph.add_node('label_bmi',lableBmi)


# add edge on graph 
graph.add_edge(START,'bmi_cal')
graph.add_edge('bmi_cal','label_bmi')
graph.add_edge("bmi_cal",END)

# compile the graph 
workflow=graph.compile()
    
final_output=workflow.invoke({"weight_kg":51,"height_m":1.64592})
print("Result:",final_output)