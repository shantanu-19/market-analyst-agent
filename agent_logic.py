from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import initialize_agent, Tool, AgentType
from tools import search_tool, get_stock_performance
from langchain_community.tools.tavily_search import TavilySearchResults
# Change this line in agent_logic.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

def get_market_agent():
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)
    
    # Define your tools exactly as before
    tools = [
        Tool(name="Web_Search", func=search_tool.run, description="Search live news"),
        Tool(name="Stock_Data", func=get_market_analysis, description="Get recent prices")
    ]
    
    # Get the standard ReAct prompt from the LangChain Hub
    # This is the "modern" way to do it for the bootcamp
    prompt = hub.pull("hwchase17/react")
    
    # Construct the ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create the executor (the engine that runs the agent)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor

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