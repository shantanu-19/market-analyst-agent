from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_classic.agents import create_react_agent, AgentExecutor
from tools import search_tool, get_market_analysis

# --- UNIVERSAL HUB IMPORT ---
try:
    from langchainhub import hub
except ImportError:
    import langchainhub as hub
# ----------------------------

def get_market_agent():
    # Brain: Llama 3.1 via NVIDIA NIM
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)
    
    # Use the tools defined in tools.py
    tools = [search_tool, get_market_analysis]
    
    # This will now work regardless of the import style
    prompt = hub.pull("hwchase17/react") 
    
    # Build the agent
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )