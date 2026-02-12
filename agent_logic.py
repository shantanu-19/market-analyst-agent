from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool # <--- Correct modern import
from langchain import hub
from tools import search_tool, get_market_analysis

def get_market_agent():
    # Initialize the NVIDIA NIM model
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)
    
    # Define tools correctly using the core Tool class
    tools = [
        Tool(
            name="Web_Search", 
            func=search_tool.run, 
            description="Search for live news explaining market events."
        ),
        Tool(
            name="Stock_Data", 
            func=get_market_analysis, 
            description="Get actual price change data for a stock symbol."
        )
    ]
    
    # Standard ReAct prompt
    prompt = hub.pull("hwchase17/react")
    
    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    
    # The Executor handles the loop
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )