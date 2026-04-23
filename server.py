"""
CallDelta MCP Server - Complete Working Version
Uses Hugging Face Inference API for sentiment (no local ML models).
Implements fallback chain and transparent materiality.
"""

import os
import json
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

app = FastAPI(title="CallDelta MCP Server")
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "3.0.0",
        "features": ["fallback_chain", "transparent_materiality", "huggingface_inference"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.post("/call")
async def handle_tool_call(request: Request):
    try:
        body = await request.json()
    except:
        body = {}
    
    tool_name = body.get("tool", "")
    arguments = body.get("arguments", {})
    
    if tool_name == "compare_earnings_calls":
        result = await compare_earnings_calls(arguments)
        return JSONResponse(content=result)
    
    elif tool_name == "analyze_sentiment":
        result = await analyze_sentiment(arguments)
        return JSONResponse(content=result)
    
    else:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown tool: {tool_name}"}
        )


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
    
    # Analyze sentiment of both transcripts
    current_sentiment = sentiment_client.analyze_sentiment(current.get('content', ''))
    previous_sentiment = sentiment_client.analyze_sentiment(previous.get('content', ''))
    
    current_score = current_sentiment.get('sentiment_score', 0.5)
    previous_score = previous_sentiment.get('sentiment_score', 0.5)
    delta = current_score - previous_score
    
    return {
        "tool": "compare_earnings_calls",
        "ticker": ticker,
        "current_quarter": f"Q{current_quarter} {current_year}",
        "previous_quarter": f"Q{previous_quarter} {previous_year}",
        "sources": {
            "current": {
                "source": current.get('source_used', 'Unknown'),
                "url": current.get('url'),
                "status": current['status']
            },
            "previous": {
                "source": previous.get('source_used', 'Unknown'),
                "url": previous.get('url'),
                "status": previous['status']
            }
        },
        "sentiment_analysis": {
            "overall_delta": {
                "current": current_score,
                "previous": previous_score,
                "delta": round(delta, 3),
                "direction": "more confident" if delta > 0.05 else ("less confident" if delta < -0.05 else "unchanged"),
                "materiality": "high" if abs(delta) > 0.15 else ("moderate" if abs(delta) > 0.08 else "low")
            },
            "current_sentiment": current_sentiment,
            "previous_sentiment": previous_sentiment,
            "methodology": {
                "model": "distilbert-base-uncased-finetuned-sst-2-english",
                "api": "HuggingFace Inference (free)",
                "sentiment_scale": "0=negative/cautious, 0.5=neutral, 1=positive/confident",
                "transparency": "All sentiment scores are backed by the Hugging Face inference API"
            }
        },
        "transparency_note": "All sentiment claims are generated using the Hugging Face inference API with the distilbert-base-uncased-finetuned-sst-2-english model.",
        "query_timestamp": datetime.now().isoformat()
    }


async def analyze_sentiment(args: dict) -> dict:
    text = args.get("text", "")
    
    if not text or len(text) < 20:
        return {
            "error": "Text is required and must be at least 20 characters",
            "user_action": "Provide an earnings call transcript excerpt or full text"
        }
    
    result = sentiment_client.analyze_sentiment(text)
    
    return {
        "tool": "analyze_sentiment",
        "analysis": result,
        "text_preview": text[:200] + "..." if len(text) > 200 else text,
        "query_timestamp": datetime.now().isoformat(),
        "transparency_note": "Sentiment analysis performed using Hugging Face inference API."
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"Health check: http://0.0.0.0:{port}/health")
    print("Using Hugging Face Inference API for sentiment (no local ML models)")
    uvicorn.run(app, host="0.0.0.0", port=port)
