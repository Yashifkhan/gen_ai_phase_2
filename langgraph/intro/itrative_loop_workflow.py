# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langgraph.graph import StateGraph ,START ,END
# from typing import TypedDict,Annotated,Literal
# from langchain_core.messages import HumanMessage, SystemMessage
# from pydantic import BaseModel, Field
# from dotenv import load_dotenv
# import operator
# load_dotenv()
# import os

# NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
# generator_llm = ChatNVIDIA(
#     # model="nvidia/nemotron-3-super-120b-a12b",
#     # model="meta/llama-3.1-8b-instruct",
#     model="nvidia/nemotron-3-ultra-550b-a55b",
#     api_key=NVIDIA_API_KEY , 
#     temperature=1,
#     top_p=1,
#     max_completion_tokens=1024,
#     seed=42,
# )
# evaluate_llm = ChatNVIDIA(
#     # model="nvidia/nemotron-3-super-120b-a12b",
#     # model="meta/llama-3.1-8b-instruct",
#     model="nvidia/nemotron-3-ultra-550b-a55b",
#     api_key=NVIDIA_API_KEY , 
#     temperature=1,
#     top_p=1,
#     max_completion_tokens=1024,
#     seed=42,
# )
# optimizer_llm = ChatNVIDIA(
#     # model="nvidia/nemotron-3-super-120b-a12b",
#     # model="meta/llama-3.1-8b-instruct",
#     model="nvidia/nemotron-3-ultra-550b-a55b",
#     api_key=NVIDIA_API_KEY , 
#     temperature=1,
#     top_p=1,
#     max_completion_tokens=1024,
#     seed=42,
# )


# class tweetState(TypedDict):
#     topic:str
#     tweet:str
#     evaluation:Literal['approved','need approved']
#     feedback:str
#     iteration:int
#     max_iteration:int
    
    
# class TweetEvaluation(BaseModel):
#     evaluation: Literal["approved", "needs_improvement"] = Field(
#         ..., description="Final evaluation result."
#     )
#     feedback: str = Field(
#         ..., description="Constructive feedback for the tweet."
#     )

# graph=StateGraph(tweetState)
# structure_evl_llm =evaluate_llm.with_structured_output(TweetEvaluation)

# def tweet_generate(state:tweetState):
#     # prompt=
#     messages = [
#     SystemMessage(
#         content="You are a funny and clever Twitter/X influencer."
#     ),
#     HumanMessage(
#         content=f"""
# Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

# Rules:
# - Do NOT use question-answer format.
# - Max 280 characters.
# - Use observational humor, irony, sarcasm, or cultural references.
# - Think in meme logic, punchlines, or relatable takes.
# - Use simple, day to day English.
# - This is version {state['iteration'] + 1}.
# """
#     )
# ]
#     response=generator_llm.invoke(messages).content
#     # state['tweet']=response
#     return {'tweet':response}

# def tweet_evaluate(state:tweetState):
#     # prompt 
#     messages = [
#     SystemMessage(
#         content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."
#     ),
#     HumanMessage(
#         content=f"""
# Evaluate the following tweet:

# Tweet: "{state['tweet']}"

# Use the criteria below to evaluate the tweet:

# 1. Originality - Is this fresh, or have you seen it a hundred times before?
# 2. Humor - Did it genuinely make you smile, laugh, or chuckle?
# 3. Punchiness - Is it short, sharp, and scroll-stopping?
# 4. Virality Potential - Would people retweet or share it?
# 5. Format - Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

# Auto-reject if:
# - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
# - It exceeds 280 characters
# - It reads like a traditional setup-punchline joke
# - Don't end with generic, throwaway, or deflating lines that weaken the humor
#   (e.g., "Masterpieces of the auntie-uncle universe" or vague summaries)

# ### Respond ONLY in structured format:
# - evaluation: "approved" or "needs_improvement"
# - feedback: One paragraph explaining the strengths and weaknesses
# """
#     )
# ]
#     response=structure_evl_llm.invoke(messages).content
#     return {"evaluation":response.evaluation,"feedback":response.feedback}


# def tweet_optimize(state:tweetState):
#     messages = [
#     SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
#     HumanMessage(content=f"""
# Improve the tweet based on this feedback:
# {state["feedback"]}

# Topic: {state["topic"]}
# Original Tweet:
# {state['tweet']}

# Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
# """)
# ]
    
#     response=optimizer_llm.invoke(messages)
#     iteration=state['itration']+1
#     return {'tweet':response,'iteration':iteration}

# def route_evel(state:tweetState):
#     if state['evaluation'] == "aproved" or state['iteration'] >= state['max_iteration']:
#         return "approved"
#     else:
#         return 'need_improvement'
    
# graph.add_node("generate", tweet_generate)
# # graph.add_node("evaluate", tweet_evaluate)
# graph.add_node("optimize", tweet_optimize)

# graph.add_edge(START, "generate")
# graph.add_edge("generate", "evaluate")

# graph.add_conditional_edges(
#     "evaluate",
#     route_evel,
#     {
#         "approved": END,
#         "needs_improvement": "optimize"
#     }
# )

# graph.add_edge("optimize", "evaluate")

# workflow=graph.compile()
# result=workflow.invoke({'topic':'ai/ml','iteration':1,'max_teration':5})
# print("final output is this : ",result)



from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# --------------------------------------------------
# LLMs
# --------------------------------------------------

generator_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=1,
    max_tokens=1024,
)

evaluate_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=1,
    max_tokens=1024,
)

optimizer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=1,
    max_tokens=1024,
)


# --------------------------------------------------
# State
# --------------------------------------------------

class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int


# --------------------------------------------------
# Structured Output Schema
# --------------------------------------------------

class TweetEvaluation(BaseModel):

    evaluation: Literal["approved", "needs_improvement"] = Field(
        ...,
        description="Final evaluation result."
    )

    feedback: str = Field(
        ...,
        description="Constructive feedback for the tweet."
    )


# Structured evaluator
structure_eval_llm = evaluate_llm.with_structured_output(
    TweetEvaluation
)


# --------------------------------------------------
# Generate Tweet
# --------------------------------------------------

def tweet_generate(state: TweetState):

    messages = [
        SystemMessage(
            content="You are a funny and clever Twitter/X influencer."
        ),

        HumanMessage(
            content=f"""
Write a short, original, and hilarious tweet on the topic:
"{state['topic']}"

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day-to-day English.
- This is version {state['iteration']}.
"""
        )
    ]

    response = generator_llm.invoke(messages).content

    return {
        "tweet": response
    }


# --------------------------------------------------
# Evaluate Tweet
# --------------------------------------------------

def tweet_evaluate(state: TweetState):

    messages = [
        SystemMessage(
            content="""
You are a ruthless, no-laugh-given Twitter critic.

Evaluate tweets based on:
- humor
- originality
- punchiness
- virality
- tweet format
"""
        ),

        HumanMessage(
            content=f"""
Evaluate the following tweet:

Tweet:
"{state['tweet']}"

Criteria:

1. Originality
   Is this fresh, or have you seen it a hundred times before?

2. Humor
   Did it genuinely make you smile, laugh, or chuckle?

3. Punchiness
   Is it short, sharp, and scroll-stopping?

4. Virality Potential
   Would people retweet or share it?

5. Format
   Is it a well-formed tweet?
   It must NOT be Q&A and must be under 280 characters.

Auto-reject if:

- It uses question-answer format.
- It exceeds 280 characters.
- It reads like a traditional setup-punchline joke.
- It ends with a generic or deflating line.

Return only:

evaluation:
"approved"

OR

evaluation:
"needs_improvement"

Also provide constructive feedback.
"""
        )
    ]

    response = structure_eval_llm.invoke(messages)

    return {
        "evaluation": response.evaluation,
        "feedback": response.feedback
    }


# --------------------------------------------------
# Optimize Tweet
# --------------------------------------------------

def tweet_optimize(state: TweetState):

    messages = [
        SystemMessage(
            content="""
You punch up tweets for virality and humor
based on the evaluator's feedback.
"""
        ),

        HumanMessage(
            content=f"""
Improve the tweet based on this feedback:

{state["feedback"]}

Topic:
{state["topic"]}

Original Tweet:
{state["tweet"]}

Rewrite it as a short, viral-worthy tweet.

Rules:
- Avoid Q&A style.
- Stay under 280 characters.
- Make it funnier.
- Make it punchier.
- Use simple, natural English.
- Do not explain the joke.
- Output ONLY the rewritten tweet.
"""
        )
    ]

    response = optimizer_llm.invoke(messages).content

    iteration = state["iteration"] + 1

    return {
        "tweet": response,
        "iteration": iteration
    }


# --------------------------------------------------
# Router
# --------------------------------------------------

def route_eval(state: TweetState):

    if (
        state["evaluation"] == "approved"
        or state["iteration"] >= state["max_iteration"]
    ):
        return "approved"

    return "needs_improvement"


# --------------------------------------------------
# Build Graph
# --------------------------------------------------

graph = StateGraph(TweetState)

graph.add_node("generate", tweet_generate)
graph.add_node("evaluate", tweet_evaluate)
graph.add_node("optimize", tweet_optimize)


# START → GENERATE
graph.add_edge(START, "generate")


# GENERATE → EVALUATE
graph.add_edge("generate", "evaluate")


# EVALUATE → APPROVED / OPTIMIZE
graph.add_conditional_edges(
    "evaluate",
    route_eval,
    {
        "approved": END,
        "needs_improvement": "optimize"
    }
)


# OPTIMIZE → EVALUATE
graph.add_edge("optimize", "evaluate")


# --------------------------------------------------
# Compile
# --------------------------------------------------

workflow = graph.compile()


# --------------------------------------------------
# Run
# --------------------------------------------------

result = workflow.invoke({
    "topic": "AI/ML",
    "tweet": "",
    "evaluation": "needs_improvement",
    "feedback": "",
    "iteration": 1,
    "max_iteration": 5
})


print("\n==============================")
print("FINAL TWEET")
print("==============================")
print(result["tweet"])

print("\n==============================")
print("EVALUATION")
print("==============================")
print(result["evaluation"])

print("\n==============================")
print("FEEDBACK")
print("==============================")
print(result["feedback"])

print("\n==============================")
print("ITERATION")
print("==============================")
print(result["iteration"])