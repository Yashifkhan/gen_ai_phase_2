from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
load_dotenv()
import os


NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
client = ChatNVIDIA(
    #   model="z-ai/glm-5.2",
    model="nvidia/nemotron-3-ultra-550b-a55b",
    #  model="nvidia/nemotron-3-super-120b-a12b",
    api_key=NVIDIA_API_KEY , 
    temperature=1,
    top_p=1,
    max_completion_tokens=16384,
    seed=42,
  
)

resp=client.invoke("what is ai/ml")
print(resp.content)

# for chunk in client.stream([{"role":"user","content":"explain me about the ai/ml"}]):
  
#     print(chunk.content, end="")
