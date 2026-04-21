"""
CallDelta MCP Server - Minimal Working Version
Remove FinBERT temporarily to get server running.
"""

import os
import json
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Create FastAPI app
app = FastAPI(title="CallDelta MCP Server")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CallDelta MCP Server",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    """Simple health check for Railway."""
    return {"status": "alive"}


@app.post("/call")
async def handle_tool_call(request: Request):
    """Handle MCP tool calls - simplified version."""
    try:
        body = await request.json()
    except:
        body = {}
    
    tool_name = body.get("tool", "")
    arguments = body.get("arguments", {})
    
    if tool_name == "compare_earnings_calls":
        ticker = arguments.get("ticker", "UNKNOWN")
        return JSONResponse(content={
            "tool": "compare_earnings_calls",
            "ticker": ticker,
            "message": "Tool is under development. Full sentiment analysis coming soon.",
            "status": "degraded",
            "timestamp": datetime.now().isoformat()
        })
    
    elif tool_name == "analyze_sentiment":
        return JSONResponse(content={
            "tool": "analyze_sentiment",
            "message": "Tool is under development. Full sentiment analysis coming soon.",
            "status": "degraded",
            "timestamp": datetime.now().isoformat()
        })
    
    else:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown tool: {tool_name}"}
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting CallDelta MCP Server on port {port}")
    print(f"Health check: http://0.0.0.0:{port}/health")
    uvicorn.run(app, host="0.0.0.0", port=port)
