import os
import langchainhub as hub # Correct 2026 hub import
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_classic.agents import create_react_agent, AgentExecutor # Fixed Import
from langchain_core.tools import Tool 
from tools import search_tool, get_market_analysis

def get_market_agent():
    # Your NVIDIA Brain
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)
    
    # Your Tools
    tools = [
        Tool(name="Web_Search", func=search_tool.run, description="Search live news."),
        Tool(name="Market_Data", func=get_market_analysis, description="Get stock trends.")
    ]
    
    # Pulling from the hub using the new 'langchainhub' driver
    prompt = hub.pull("hwchase17/react")
    
    # Logic built with classic compatibility
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )