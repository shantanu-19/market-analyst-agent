import streamlit as st
from agent_logic import get_market_agent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="NVIDIA Analyst", layout="wide")
st.title("🤖 Intelligent Market Analyst")

symbol = st.text_input("Enter Ticker:", "NVDA")

if st.button("Run Research"):
    with st.spinner("Agent is working..."):
        try:
            agent = get_market_agent()
            response = agent.invoke({"input": f"Analyze {symbol} price and news."})
            st.write(response["output"])
        except Exception as e:
            st.error(f"Error: {e}")