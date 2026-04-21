"""
CallDelta MCP Server - Full Implementation
Implements fallback chain and transparent materiality.
"""

import os
import json
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from transcript_fetcher import TranscriptFetcher
from sentiment_analyzer import TransparentSentimentAnalyzer

app = FastAPI(title="CallDelta MCP Server")
fetcher = TranscriptFetcher()
sentiment_analyzer = TransparentSentimentAnalyzer()


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "2.0.0",
        "features": ["fallback_chain", "transparent_materiality"],
        "timestamp": datetime.now().isoformat()
    }


@app.post("/call")
async def handle_tool_call(request: Request):
    body = await request.json()
    tool_name = body.get("tool")
    arguments = body.get("arguments", {})
    
    if tool_name == "compare_earnings_calls":
        result = await compare_earnings_calls(arguments)
        return JSONResponse(content=result)
    else:
        return JSONResponse(status_code=400, content={"error": f"Unknown tool: {tool_name}"})


async def compare_earnings_calls(args: dict) -> dict:
    ticker = args.get("ticker", "").upper()
    current_year = args.get("current_year")
    current_quarter = args.get("current_quarter")
    previous_year = args.get("previous_year")
    previous_quarter = args.get("previous_quarter")
    
    if not ticker:
        return {"error": "Ticker is required"}
    
    # Fetch current transcript
    current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
    
    if current['status'] == 'error':
        return {
            "error": "Failed to fetch current transcript",
            "source_error": current,
            "user_action": f"Transcript for {ticker} Q{current_quarter} {current_year} is not available. Try a different ticker or quarter."
        }
    
    # Fetch previous transcript
    previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
    
    if previous['status'] == 'error':
        return {
            "error": "Failed to fetch previous transcript",
            "source_error": previous,
            "user_action": f"Transcript for {ticker} Q{previous_quarter} {previous_year} is not available. Try a different ticker or quarter."
        }
    
    # Compare sentiment
    comparison = sentiment_analyzer.compare_transcripts(
        current.get('content', ''),
        previous.get('content', '')
    )
    
    return {
        "tool": "compare_earnings_calls",
        "ticker": ticker,
        "current_quarter": f"Q{current_quarter} {current_year}",
        "previous_quarter": f"Q{previous_quarter} {previous_year}",
        "sources": {
            "current": {
                "source": current.get('source_used', current.get('source', 'Unknown')),
                "url": current.get('url'),
                "status": current['status']
            },
            "previous": {
                "source": previous.get('source_used', previous.get('source', 'Unknown')),
                "url": previous.get('url'),
                "status": previous['status']
            }
        },
        "sentiment_analysis": comparison,
        "transparency_note": "All sentiment claims are backed by exact source sentences and model outputs. No black-box verdicts.",
        "query_timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
