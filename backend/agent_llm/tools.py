"""
agent_llm/tools.py — All LangChain tools bound to the chat LLM.

Tools:
  1. weather_tool   — wttr.in (no API key)
  2. currency_tool  — exchangerate-api.com (free tier)
  3. tavily_search  — Tavily live web search
  4. stock_tool     — Alpha Vantage real-time quotes
  5. wikipedia_tool — Wikipedia lookup
"""

import httpx
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langsmith import traceable

from backend.config import get_settings

settings = get_settings()


# ── 1. Weather ────────────────────────────────────────────────────────────────

@tool
@traceable(name="weather_tool")
async def weather_tool(location: str) -> str:
    """
    Get current weather for any location.
    Input: city name e.g. 'London' or 'Paris, France' or 'Lahore, Pakistan'
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"https://wttr.in/{location}?format=j1")
            data = resp.json()

        cur     = data["current_condition"][0]
        area    = data["nearest_area"][0]
        city    = area["areaName"][0]["value"]
        country = area["country"][0]["value"]

        return (
            f"Weather in {city}, {country}:\n"
            f"  Temp:      {cur['temp_C']}°C / {cur['temp_F']}°F\n"
            f"  Feels like:{cur['FeelsLikeC']}°C\n"
            f"  Condition: {cur['weatherDesc'][0]['value']}\n"
            f"  Humidity:  {cur['humidity']}%\n"
            f"  Wind:      {cur['windspeedKmph']} km/h"
        )
    except Exception as e:
        return f"Could not fetch weather for '{location}': {e}"


# ── 2. Currency ───────────────────────────────────────────────────────────────

@tool
@traceable(name="currency_tool")
async def currency_tool(query: str) -> str:
    """
    Convert currency or get exchange rates.
    Input format: 'USD to EUR' or '100 USD to PKR'
    Examples: 'USD to EUR', '500 GBP to JPY', '1000 PKR to USD'
    """
    try:
        parts = query.upper().split()
        if len(parts) == 3:
            from_cur, _, to_cur = parts
            amount = 1.0
        elif len(parts) == 4:
            amount, from_cur, _, to_cur = parts
            amount = float(amount)
        else:
            return "Invalid format. Use: 'USD to EUR' or '100 USD to EUR'"

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"https://api.exchangerate-api.com/v4/latest/{from_cur}")
            data = resp.json()

        rate = data.get("rates", {}).get(to_cur)
        if rate is None:
            return f"Currency {to_cur} not found."

        result = amount * rate
        return (
            f"{amount} {from_cur} = {result:.4f} {to_cur}\n"
            f"Rate: 1 {from_cur} = {rate:.4f} {to_cur}\n"
            f"Updated: {data.get('date', 'N/A')}"
        )
    except Exception as e:
        return f"Currency conversion error: {e}"


# ── 3. Tavily Web Search ──────────────────────────────────────────────────────

@traceable(name="tavily_search_tool")
def _build_tavily():
    return TavilySearchResults(
        max_results    = 3,
        tavily_api_key = settings.tavily_api_key,
        description    = (
            "Search the web for current news, recent events, and real-time information. "
            "Use for anything that requires up-to-date facts beyond training data."
        ),
    )

tavily_search = _build_tavily()


# ── 4. Stock / Finance ────────────────────────────────────────────────────────

@tool
@traceable(name="stock_tool")
async def stock_tool(symbol: str) -> str:
    """
    Get real-time stock quote for any publicly traded company.
    Input: ticker symbol e.g. 'AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN'
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "GLOBAL_QUOTE",
                    "symbol":   symbol.upper(),
                    "apikey":   settings.alpha_vantage_api_key,
                },
            )
            data = resp.json()

        quote = data.get("Global Quote", {})
        if not quote:
            return f"No data for '{symbol}'. Check the ticker symbol."

        price      = float(quote.get("05. price", 0))
        change     = float(quote.get("09. change", 0))
        change_pct = quote.get("10. change percent", "0%")
        volume     = int(quote.get("06. volume", 0))
        high       = float(quote.get("03. high", 0))
        low        = float(quote.get("04. low", 0))
        prev_close = float(quote.get("08. previous close", 0))
        day        = quote.get("07. latest trading day", "N/A")
        arrow      = "▲" if change >= 0 else "▼"

        return (
            f"{symbol.upper()} — {day}\n"
            f"  Price:      ${price:.2f}  {arrow} {change:+.2f} ({change_pct})\n"
            f"  Day Range:  ${low:.2f} – ${high:.2f}\n"
            f"  Prev Close: ${prev_close:.2f}\n"
            f"  Volume:     {volume:,}"
        )
    except Exception as e:
        return f"Could not fetch stock data for '{symbol}': {e}"


# ── 5. Wikipedia ──────────────────────────────────────────────────────────────

@traceable(name="wikipedia_tool")
def _build_wikipedia():
    return WikipediaQueryRun(
        api_wrapper = WikipediaAPIWrapper(
            top_k_results        = 2,
            doc_content_chars_max= 1500,
        ),
        description = (
            "Look up encyclopedic facts, definitions, historical events, "
            "scientific concepts, or background knowledge on any topic."
        ),
    )

wikipedia_tool = _build_wikipedia()


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    weather_tool,
    currency_tool,
    tavily_search,
    stock_tool,
    wikipedia_tool,
]