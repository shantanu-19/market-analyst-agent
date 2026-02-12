from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain import hub
from tools import search_tool, get_market_analysis

def get_market_agent():
    # 1. Initialize the NVIDIA "Brain"
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)
    
    # 2. Package tools for the agent
    # We wrap our functions in Tool objects so the agent knows how to use them
    tools = [
        Tool(
            name="Web_Search", 
            func=search_tool.run, 
            description="Use this for live news and explaining market events."
        ),
        Tool(
            name="Stock_Data", 
            func=get_market_analysis, 
            description="Use this to get actual price changes and trends."
        )
    ]
    
    # 3. Pull the standard reasoning prompt (ReAct)
    # This tells the model how to "Think", "Act", and "Observe"
    prompt = hub.pull("hwchase17/react")
    
    # 4. Construct the Agent logic
    agent = create_react_agent(llm, tools, prompt)
    
    # 5. Create the Executor to run the loop
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )