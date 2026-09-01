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

generator_llm=model()
evaluate_llm=model()
optimizer_llm=model()

class tweetState(TypedDict):
    topic:str
    tweet:str
    evaluation:Literal['approved','need approved']
    feedback:str
    iteration:int
    max_iteration:int
    
    
class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        ..., description="Final evaluation result."
    )
    feedback: str = Field(
        ..., description="Constructive feedback for the tweet."
    )

graph=StateGraph(tweetState)
structure_evl_llm =evaluate_llm.with_structure_output(TweetEvaluation)

def tweet_generate(state:tweetState):
    # prompt=
    messages = [
    SystemMessage(
        content="You are a funny and clever Twitter/X influencer."
    ),
    HumanMessage(
        content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day English.
- This is version {state['iteration'] + 1}.
"""
    )
]
    response=generator_llm(messages).content
    # state['tweet']=response
    return {'tweet',response}

def tweet_evalaute(state:tweetState):
    # prompt 
    messages = [
    SystemMessage(
        content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."
    ),
    HumanMessage(
        content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"

Use the criteria below to evaluate the tweet:

1. Originality - Is this fresh, or have you seen it a hundred times before?
2. Humor - Did it genuinely make you smile, laugh, or chuckle?
3. Punchiness - Is it short, sharp, and scroll-stopping?
4. Virality Potential - Would people retweet or share it?
5. Format - Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Don't end with generic, throwaway, or deflating lines that weaken the humor
  (e.g., "Masterpieces of the auntie-uncle universe" or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"
- feedback: One paragraph explaining the strengths and weaknesses
"""
    )
]
    response=structure_evl_llm(messages).content
    return {"evaluation":response.evaluation,"feedback":response.feedback}


def tweet_optimize(state:tweetState):
    
graph.add_node("generate",tweet_generate)    
graph.add_node("evalaute",tweet_evalaute)    
graph.add_node("optimize",tweet_optimize)    