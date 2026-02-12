import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain import hub
from tools import search_tool, get_market_analysis

def get_market_agent():
    # 1. The Brain: Llama 3.1
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)
    
    # 2. The Memory/Retriever: Llama 3.2 NeMo Retriever (Your choice!)
    # Even if not used for a database yet, having this ready is great for the bootcamp
    embedder = NVIDIAEmbeddings(model="nvidia/llama-3_2-nemoretriever-300m-embed-v2")

    # 3. Packaging Tools
    tools = [
        Tool(
            name="Web_Search", 
            func=search_tool.run, 
            description="Use for current news and explaining price changes."
        ),
        Tool(
            name="Market_Data", 
            func=get_market_analysis, 
            description="Get price trends and percentages for a stock ticker."
        )
    ]
    
    # 4. Pull reasoning prompt
    prompt = hub.pull("hwchase17/react")
    
    # 5. Build Agent
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )