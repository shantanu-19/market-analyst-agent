import yfinance as yf
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool

# 1. Optimized Search Tool
# k=5 gives the agent more context to compare different news sources
# search_depth="advanced" is the "Agentic" way to get deep reasoning
# topic="finance" filters out noise (e.g., celebrity news about NVIDIA)
search_tool = TavilySearchResults(
    k=5, 
    search_depth="advanced", 
    topic="finance"
)

# 2. Refined Financial Analysis Tool
@tool
def get_market_analysis(symbol: str):
    """
    Fetches the last 5 days of stock data for a given symbol. 
    Use this to identify price trends and volatility before searching for news.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        
        if hist.empty:
            return f"No data found for symbol: {symbol}. Ensure you are using a valid ticker (e.g., 'NVDA' for NVIDIA)."
        
        # Calculate key metrics for the agent to reason with
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        pct_change = ((end_price - start_price) / start_price) * 100
        volatility = hist['High'].max() - hist['Low'].min()
        
        return {
            "symbol": symbol,
            "current_price": round(end_price, 2),
            "5d_percent_change": f"{pct_change:.2f}%",
            "volatility_range": round(volatility, 2),
            "summary": f"The stock has moved {pct_change:.2f}% over the last 5 days."
        }
    except Exception as e:
        return f"System Error: Could not retrieve data for {symbol}. Error: {str(e)}"