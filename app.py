import streamlit as st
from agent_logic import get_market_agent

st.title("🤖 Intelligent Market Analyst")
st.markdown("Enter a stock symbol to trigger an autonomous investigation.")

symbol = st.text_input("Stock Symbol (e.g., NVDA, AAPL)", "NVDA")

if st.button("Run Autonomously"):
    with st.spinner(f"Agent is researching {symbol}..."):
        agent = get_market_agent()
        # The prompt that triggers the agentic loop
        prompt = (f"Check the recent performance of {symbol}. "
                  f"If there are significant moves, search the web to explain why. "
                  f"Provide a summary of the 'Reason for Movement'.")
        
        response = agent.run(prompt)
        st.subheader("Agent's Findings")
        st.write(response)