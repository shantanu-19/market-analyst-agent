from langchain_nvidia_ai_endpoints import ChatNVIDIA
# Import from classic to fix the Streamlit Cloud error
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool 
from langchain import hub
from tools import search_tool, get_market_analysis

def get_market_agent():
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)
    
    tools = [
        Tool(name="Web_Search", func=search_tool.run, description="Search live news."),
        Tool(name="Market_Data", func=get_market_analysis, description="Get stock trends.")
    ]
    
    prompt = hub.pull("hwchase17/react")
    
    # This now uses the stable legacy engine via langchain-classic
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )