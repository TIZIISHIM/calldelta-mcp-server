"""
Transcript Fetcher using Financial Modeling Prep API
No scraping - reliable data source with free tier.
"""

import os
import requests
from typing import Dict
from datetime import datetime

class TranscriptFetcher:
    def __init__(self):
        self.api_key = os.environ.get("FMP_API_KEY", "")
        self.base_url = "https://financialmodelingprep.com/api/v3"
    
    def fetch_transcript(self, ticker: str, year: int, quarter: int) -> Dict:
        """
        Fetch transcript from Financial Modeling Prep API.
        This works reliably - no scraping, no rate limits on free tier.
        """
        if not self.api_key:
            return {
                'status': 'error',
                'error_code': 'MISSING_API_KEY',
                'error_message': 'FMP_API_KEY environment variable not set. Get a free key at https://financialmodelingprep.com',
                'timestamp': datetime.now().isoformat()
            }
        
        quarter_map = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
        quarter_str = quarter_map.get(quarter, f'Q{quarter}')
        
        # Format: Q3 2024
        quarter_param = f"{quarter_str} {year}"
        
        url = f"{self.base_url}/earning_call_transcript/{ticker}"
        params = {
            'quarter': quarter_param,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    transcript_content = data[0].get('content', '')
                    if transcript_content and len(transcript_content) > 200:
                        return {
                            'status': 'success',
                            'source': 'Financial Modeling Prep',
                            'content': transcript_content,
                            'url': f"https://financialmodelingprep.com/earnings-call-transcript/{ticker}/{quarter_param}",
                            'source_used': 'FMP API',
                            'content_length': len(transcript_content),
                            'timestamp': datetime.now().isoformat()
                        }
            
            # Transcript not found
            return {
                'status': 'error',
                'error_code': 'NOT_FOUND',
                'error_message': f"No transcript found for {ticker} {quarter_str} {year}",
                'suggestion': f"Try recent quarters like Q2 2024 or Q1 2024. Transcripts are usually available 2-3 days after earnings.",
                'ticker': ticker,
                'year': year,
                'quarter': quarter,
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.Timeout:
            return {
                'status': 'error',
                'error_code': 'TIMEOUT',
                'error_message': 'FMP API request timed out',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_code': 'UNKNOWN',
                'error_message': f'API error: {str(e)[:100]}',
                'timestamp': datetime.now().isoformat()
            }
