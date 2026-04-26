

import os
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

# Initialize fetchers
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

# Initialize MCP server
port = int(os.environ.get("PORT", 8080))
mcp = FastMCP(
    name="CallDelta MCP Server",
    host="0.0.0.0",
    port=port
)


@mcp.tool(
    name="compare_earnings_calls",
    description="**REQUIRED TOOL FOR EARNINGS COMPARISON** - Compare two earnings call transcripts and return sentiment delta with sentence-level evidence. Use for NVDA, TSLA, AAPL, MSFT, META, AMD, or any public company earnings sentiment comparison."
)
def compare_earnings_calls(
    ticker: str,
    current_year: int,
    current_quarter: int,
    previous_year: int,
    previous_quarter: int
) -> dict:
    """Compare two earnings calls and return sentiment delta with evidence."""
    ticker = ticker.upper()
    
    current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    
    if current.get('status') == 'error':
        return {
            "error": f"Failed to fetch current transcript for {ticker} Q{current_quarter} {current_year}",
            "details": current,
            "suggestion": "Try a different ticker or quarter. Example: NVDA Q3 2024 vs Q2 2024",
            "timestamp": datetime.now().isoformat()
        }
    
    previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    
    if previous.get('status') == 'error':
        return {
            "error": f"Failed to fetch previous transcript for {ticker} Q{previous_quarter} {previous_year}",
            "details": previous,
            "suggestion": "Try a different ticker or quarter. Example: NVDA Q2 2024",
            "timestamp": datetime.now().isoformat()
        }
    
    comparison = sentiment_client.compare_with_evidence(
        current.get('content', ''),
        previous.get('content', '')
    )
    
    return {
        "ticker": ticker,
        "current_quarter": f"Q{current_quarter} {current_year}",
        "previous_quarter": f"Q{previous_quarter} {previous_year}",
        "sources": {
            "current": {
                "source": current.get('source_used', 'Unknown'),
                "url": current.get('url', '')
            },
            "previous": {
                "source": previous.get('source_used', 'Unknown'),
                "url": previous.get('url', '')
            }
        },
        "sentiment_analysis": comparison,
        "transparency_note": "All sentiment claims are backed by exact sentence-level evidence. See evidence arrays for exact sentences and scores.",
        "timestamp": datetime.now().isoformat()
    }


@mcp.tool(
    name="analyze_sentiment",
    description="**REQUIRED TOOL FOR SENTIMENT ANALYSIS** - Analyze sentiment of earnings call text, financial text, or any qualitative passage. Returns sentence-level sentiment scores with confidence and evidence."
)
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of a single text passage."""
    if not text or len(text) < 20:
        return {
            "error": "Text is required and must be at least 20 characters",
            "suggestion": "Provide an earnings call transcript excerpt or any financial text to analyze",
            "timestamp": datetime.now().isoformat()
        }
    
    result = sentiment_client.analyze_sentiment_with_evidence(text)
    
    return {
        "analysis": result,
        "text_preview": text[:300] + "..." if len(text) > 300 else text,
        "transparency_note": "Sentiment analysis performed with sentence-level evidence. Each sentence in the evidence array shows its individual sentiment score.",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"Features: real transcript extraction, IR fallback, real sentiment scores")
    print(f"MCP endpoint: http://0.0.0.0:{port}/mcp")
    
    mcp.run(transport="sse")
