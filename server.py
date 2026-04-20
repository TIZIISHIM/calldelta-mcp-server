"""
CallDelta MCP Server - Earnings Call Transcript Delta Intelligence
Implements transparent materiality and source resilience.
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Optional

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from transcript_fetcher import TranscriptFetcher
from sentiment_analyzer import TransparentSentimentAnalyzer
from delta_detector import TranscriptDeltaDetector

# Initialize server
server = Server("calldelta-mcp-server")

# Initialize components
fetcher = TranscriptFetcher()
sentiment_analyzer = TransparentSentimentAnalyzer()
delta_detector = TranscriptDeltaDetector()


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="compare_earnings_calls",
            description="Compare two earnings call transcripts (current quarter vs previous quarter) and return structured delta showing changes in management tone, guidance language, and topic-specific sentiment shifts. Every claim includes exact source sentences, FinBERT output, and confidence scores. No black-box verdicts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'NVDA', 'TSLA', 'AAPL')"
                    },
                    "current_year": {
                        "type": "integer",
                        "description": "Year of the current earnings call"
                    },
                    "current_quarter": {
                        "type": "integer",
                        "description": "Quarter number of the current earnings call (1, 2, 3, or 4)"
                    },
                    "previous_year": {
                        "type": "integer",
                        "description": "Year of the previous earnings call"
                    },
                    "previous_quarter": {
                        "type": "integer",
                        "description": "Quarter number of the previous earnings call (1, 2, 3, or 4)"
                    }
                },
                "required": ["ticker", "current_year", "current_quarter", "previous_year", "previous_quarter"]
            }
        ),
        types.Tool(
            name="analyze_sentiment",
            description="Analyze sentiment of an earnings call transcript or text passage. Returns transparent results with sentence-level evidence, FinBERT output, and confidence scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text passage to analyze"
                    }
                },
                "required": ["text"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    """Handle tool execution."""
    
    if name == "compare_earnings_calls":
        ticker = arguments.get("ticker", "").upper()
        current_year = arguments.get("current_year")
        current_quarter = arguments.get("current_quarter")
        previous_year = arguments.get("previous_year")
        previous_quarter = arguments.get("previous_quarter")
        
        # Validate inputs
        if not ticker:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "Ticker is required"}, indent=2)
            )]
        
        # Fetch current transcript
        current_result = fetcher.fetch_transcript(ticker, current_year, current_quarter)
        
        if current_result['status'] == 'error':
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "error": "Failed to fetch transcript",
                    "details": current_result,
                    "user_action": f"Transcript for {ticker} Q{current_quarter} {current_year} is not available. Try a different ticker or quarter."
                }, indent=2)
            )]
        
        # Fetch previous transcript
        previous_result = fetcher.fetch_transcript(ticker, previous_year, previous_quarter)
        
        if previous_result['status'] == 'error':
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "error": "Failed to fetch previous transcript",
                    "details": previous_result,
                    "user_action": f"Transcript for {ticker} Q{previous_quarter} {previous_year} is not available. Try a different ticker or quarter."
                }, indent=2)
            )]
        
        # Compare transcripts using both methods
        sentiment_comparison = sentiment_analyzer.compare_transcripts(
            current_result['content'],
            previous_result['content']
        )
        
        text_changes = delta_detector.detect_changes(
            current_result['content'],
            previous_result['content']
        )
        
        # Build response with full transparency
        response = {
            'tool': 'compare_earnings_calls',
            'ticker': ticker,
            'current_quarter': f"Q{current_quarter} {current_year}",
            'previous_quarter': f"Q{previous_quarter} {previous_year}",
            'sources': {
                'current': {
                    'source': current_result['source'],
                    'url': current_result.get('url'),
                    'fetched_at': current_result.get('timestamp')
                },
                'previous': {
                    'source': previous_result['source'],
                    'url': previous_result.get('url'),
                    'fetched_at': previous_result.get('timestamp')
                }
            },
            'sentiment_analysis': sentiment_comparison,
            'text_changes': text_changes,
            'transparency_note': 'All sentiment claims are backed by exact source sentences and FinBERT model outputs. No black-box verdicts.',
            'methodology': {
                'sentiment_model': 'FinBERT (ProsusAI/finbert)',
                'change_detection': 'difflib.SequenceMatcher with paragraph-level comparison',
                'fallback_chain': 'Seeking Alpha → Fool.com → IR Page'
            },
            'query_timestamp': datetime.now().isoformat()
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]
    
    elif name == "analyze_sentiment":
        text = arguments.get("text", "")
        
        if not text or len(text) < 50:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "error": "Text is required and must be at least 50 characters",
                    "user_action": "Provide an earnings call transcript excerpt or full text"
                }, indent=2)
            )]
        
        analysis = sentiment_analyzer.analyze_transcript(text)
        
        response = {
            'tool': 'analyze_sentiment',
            'analysis': analysis,
            'transparency_note': 'All sentiment claims include the exact sentences that drove the score and the FinBERT model output.',
            'query_timestamp': datetime.now().isoformat()
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]
    
    else:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2)
        )]


async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="calldelta-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
