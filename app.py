

import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse, Response
from ctxprotocol import create_context_middleware
import uvicorn

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

# Initialize
app = FastAPI(title="CallDelta MCP Server")
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

# Create auth middleware
verify_context = create_context_middleware(
    audience="https://calldelta-mcp-server-production.up.railway.app"
)

# Define tools (plain Python dict, no mcp package needed)
TOOLS = [
    {
        "name": "compare_earnings_calls",
        "description": "Compare two earnings call transcripts and return sentiment delta.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "current_year": {"type": "integer"},
                "current_quarter": {"type": "integer"},
                "previous_year": {"type": "integer"},
                "previous_quarter": {"type": "integer"}
            },
            "required": ["ticker", "current_year", "current_quarter", "previous_year", "previous_quarter"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "current_quarter": {"type": "string"},
                "previous_quarter": {"type": "string"},
                "sentiment_analysis": {"type": "object"},
                "timestamp": {"type": "string"}
            }
        }
    },
    {
        "name": "analyze_sentiment",
        "description": "Analyze sentiment of earnings call text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "object"},
                "timestamp": {"type": "string"}
            }
        }
    }
]


@app.get("/health")
async def health():
    return {"status": "alive"}


@app.post("/mcp")
async def mcp_endpoint(request: Request, context: dict = Depends(verify_context)):
    """Handle MCP JSON-RPC requests."""
    try:
        body = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
    
    method = body.get("method")
    msg_id = body.get("id")
    params = body.get("params", {})
    
    print(f"Method: {method}, ID: {msg_id}")
    
    # initialize
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "calldelta", "version": "1.0.0"}
            }
        }
    
    # initialized notification
    if method == "notifications/initialized":
        return Response(status_code=202)
    
    # tools/list
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOLS}
        }
    
    # tools/call
    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "compare_earnings_calls":
            result = await compare_earnings_calls(args)
        elif tool_name == "analyze_sentiment":
            result = await analyze_sentiment(args)
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}
            }
        
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result)}]
            }
        }
    
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"}
    }


async def compare_earnings_calls(args):
    ticker = args.get("ticker", "").upper()
    current = fetcher.fetch_transcript(ticker, args.get("current_year"), args.get("current_quarter"))
    previous = fetcher.fetch_transcript(ticker, args.get("previous_year"), args.get("previous_quarter"))
    
    if current.get('status') == 'error':
        return {"error": "Current transcript not found"}
    if previous.get('status') == 'error':
        return {"error": "Previous transcript not found"}
    
    comparison = sentiment_client.compare_with_evidence(current.get('content', ''), previous.get('content', ''))
    
    return {
        "ticker": ticker,
        "current_quarter": f"Q{args.get('current_quarter')} {args.get('current_year')}",
        "previous_quarter": f"Q{args.get('previous_quarter')} {args.get('previous_year')}",
        "sentiment_analysis": comparison,
        "timestamp": datetime.now().isoformat()
    }


async def analyze_sentiment(args):
    text = args.get("text", "")
    if len(text) < 20:
        return {"error": "Text too short"}
    return {
        "analysis": sentiment_client.analyze_sentiment_with_evidence(text),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
