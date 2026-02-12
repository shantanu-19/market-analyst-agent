import streamlit as st
from agent_logic import get_market_agent

st.set_page_config(page_title="Intelligent Market Analyst", layout="wide")
st.title("🤖 NVIDIA Agentic Analyst")

symbol = st.text_input("Enter Ticker (e.g., NVDA, BTC-USD):", "NVDA")

if st.button("Start Autonomous Research"):
    agent_executor = get_market_agent()
    
    with st.chat_message("assistant"):
        st_callback = st.container() # To show thoughts live
        response = agent_executor.invoke({
            "input": f"Analyze the recent price movement for {symbol}. Find the news that explains why it moved."
        })
        st.write(response["output"])