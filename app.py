"""
CallDelta MCP Server - CORRECT IMPLEMENTATION
Uses mcp package for server, ctxprotocol for auth.
"""

import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from ctxprotocol import create_context_middleware
from mcp.server import Server
from mcp.types import Tool, TextContent
import uvicorn

from transcript_fetcher import TranscriptFetcher
from huggingface_client import HuggingFaceClient

# Initialize FastAPI app
app = FastAPI(title="CallDelta MCP Server")

# Create MCP server instance (THIS IS WHAT WAS MISSING BEFORE)
mcp_server = Server("calldelta-mcp-server")

# Initialize fetchers
fetcher = TranscriptFetcher()
sentiment_client = HuggingFaceClient()

# Define the output schema for tools
COMPARE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "current_quarter": {"type": "string"},
        "previous_quarter": {"type": "string"},
        "sources": {"type": "object"},
        "sentiment_analysis": {"type": "object"},
        "transparency_note": {"type": "string"},
        "timestamp": {"type": "string"}
    }
}

ANALYZE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "object"},
        "transparency_note": {"type": "string"},
        "timestamp": {"type": "string"}
    }
}


# Register tools with the MCP server
@mcp_server.list_tools()
async def list_tools():
    """Return the list of available tools."""
    return [
        Tool(
            name="compare_earnings_calls",
            description="Compare two earnings call transcripts and return sentiment delta with sentence-level evidence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., NVDA, TSLA, AAPL, MSFT)"},
                    "current_year": {"type": "integer", "description": "Year of current earnings call"},
                    "current_quarter": {"type": "integer", "description": "Quarter number (1-4)"},
                    "previous_year": {"type": "integer", "description": "Year of previous earnings call"},
                    "previous_quarter": {"type": "integer", "description": "Quarter number (1-4)"}
                },
                "required": ["ticker", "current_year", "current_quarter", "previous_year", "previous_quarter"]
            },
            outputSchema=COMPARE_OUTPUT_SCHEMA,
            _meta={"surface": "query", "queryEligible": True}
        ),
        Tool(
            name="analyze_sentiment",
            description="Analyze sentiment of earnings call text with sentence-level evidence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze for sentiment"}
                },
                "required": ["text"]
            },
            outputSchema=ANALYZE_OUTPUT_SCHEMA,
            _meta={"surface": "query", "queryEligible": True}
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Execute a tool."""
    
    if name == "compare_earnings_calls":
        ticker = arguments.get("ticker", "").upper()
        current_year = arguments.get("current_year")
        current_quarter = arguments.get("current_quarter")
        previous_year = arguments.get("previous_year")
        previous_quarter = arguments.get("previous_quarter")
        
        if not ticker:
            return [TextContent(type="text", text=json.dumps({"error": "Ticker is required"}))]
        
        current = fetcher.fetch_transcript(ticker, current_year, current_quarter)
        if current.get('status') == 'error':
            return [TextContent(type="text", text=json.dumps({"error": f"Failed to fetch transcript for {ticker} Q{current_quarter} {current_year}", "details": current}))]
        
        previous = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
        if previous.get('status') == 'error':
            return [TextContent(type="text", text=json.dumps({"error": f"Failed to fetch transcript for {ticker} Q{previous_quarter} {previous_year}", "details": previous}))]
        
        comparison = sentiment_client.compare_with_evidence(
            current.get('content', ''),
            previous.get('content', '')
        )
        
        result = {
            "ticker": ticker,
            "current_quarter": f"Q{current_quarter} {current_year}",
            "previous_quarter": f"Q{previous_quarter} {previous_year}",
            "sources": {
                "current": {"source": current.get('source_used', 'Unknown')},
                "previous": {"source": previous.get('source_used', 'Unknown')}
            },
            "sentiment_analysis": comparison,
            "transparency_note": "All claims backed by sentence-level evidence.",
            "timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "analyze_sentiment":
        text = arguments.get("text", "")
        if len(text) < 20:
            return [TextContent(type="text", text=json.dumps({"error": "Text must be at least 20 characters"}))]
        
        result = sentiment_client.analyze_sentiment_with_evidence(text)
        output = {
            "analysis": result,
            "transparency_note": "Sentence-level evidence provided.",
            "timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(type="text", text=json.dumps(output, indent=2))]
    
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# Create Context auth middleware
verify_context = create_context_middleware(
    audience="https://calldelta-mcp-server-production.up.railway.app"
)


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "17.0.0",
        "features": ["mcp_package", "context_auth_middleware", "outputSchema", "_meta"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.post("/mcp")
async def mcp_endpoint(request: Request, context: dict = Depends(verify_context)):
    """
    MCP endpoint that uses the mcp package's dispatch_request.
    This handles initialize, tools/list, and tools/call automatically.
    """
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {str(e)}"}}
        )
    
    # Let the MCP server dispatch the request
    response = await mcp_server.dispatch_request(body)
    return JSONResponse(content=response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"MCP endpoint: http://0.0.0.0:{port}/mcp")
    print(f"Health check: http://0.0.0.0:{port}/health")
    print("Using mcp package for server, ctxprotocol for auth")
    uvicorn.run(app, host="0.0.0.0", port=port)
