import yfinance as yf
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper # <--- ADD THIS
from langchain.tools import tool

# 1. Create the Wrapper first
# This is where the parameters k, search_depth, and topic now live
tavily_wrapper = TavilySearchAPIWrapper(
    k=5, 
    search_depth="advanced", 
    topic="finance"
)

# 2. Initialize the Tool using the wrapper
search_tool = TavilySearchResults(api_wrapper=tavily_wrapper)

@tool
def get_market_analysis(symbol: str):
    """
    Fetches the last 5 days of stock data for a given symbol. 
    Use this to identify price trends before searching for news.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            return f"No data found for {symbol}."
        
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        pct_change = ((end_price - start_price) / start_price) * 100
        
        return {
            "symbol": symbol,
            "pct_change": f"{pct_change:.2f}%",
            "last_price": round(end_price, 2),
        }
    except Exception as e:
        return f"Error: {str(e)}"