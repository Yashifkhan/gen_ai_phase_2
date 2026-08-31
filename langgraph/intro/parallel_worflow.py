from langgraph.graph import StateGraph,START ,END
from typing import TypedDict

# define the state

class BatsManState(TypedDict):
    runs:int
    balls:int
    fours:int
    sixes:int 
    
    sr:float
    bpb:float
    boundary_per:float  

graph=StateGraph(BatsManState)

def cal_sr(state:BatsManState):
    sr=(state['runs']/state['balls'])*100
    state['sr']=sr
    return {'sr':sr}
def cal_bpb(state:BatsManState):
    bpb=state['balls']/(state['fours'] + state['sixes'])
    state['bpb']=bpb
    return {'bpb':bpb}

def cal_boundry_per(state:BatsManState):
    boundary_per=(((state['fours']*4 +state['sixes']*6))/state['runs'])*100
    state['boundary_per']=boundary_per
    return {'boundary_per':boundary_per}

def cal_summary(state:BatsManState):
    summary= f" final summary of result: {state['sr']} \n {state['bpb']} \n {state['boundary_per']}"
    state['summary']=summary
    return {'summary':summary}
    

graph.add_node("cal_sr",cal_sr)
graph.add_node("cal_bpb",cal_bpb)
graph.add_node("cal_boundry_per",cal_boundry_per)
graph.add_node("cal_summary",cal_summary)

graph.add_edge(START,'cal_sr')
graph.add_edge(START,'cal_bpb')
graph.add_edge(START,'cal_boundry_per')

graph.add_edge('cal_sr','cal_summary')
graph.add_edge('cal_bpb','cal_summary')
graph.add_edge('cal_boundry_per','cal_summary')

graph.add_edge('cal_summary',END)

workflow=graph.compile()
result=workflow.invoke({'runs':100,'balls':50,'fours':6,'sixes':4})
print(result)