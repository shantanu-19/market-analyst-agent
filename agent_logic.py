from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import initialize_agent, Tool, AgentType
from tools import search_tool, get_stock_performance
from langchain_community.tools.tavily_search import TavilySearchResults

# This is what you "take" from Tavily to give to your NVIDIA model
search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced", 
    topic="finance"
)

def get_market_agent():
    # Using NVIDIA NIM for high-performance reasoning
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)
    
    tools = [
        Tool(name="Web_Search", func=search_tool.run, description="Search live news"),
        Tool(name="Stock_Data", func=get_stock_performance, description="Get recent prices")
    ]
    
    agent = initialize_agent(
        tools, llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )
    return agent