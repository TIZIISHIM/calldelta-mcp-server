import os
import requests
from typing import Dict
from datetime import datetime

class TranscriptFetcher:
    def __init__(self):
        self.fmp_api_key = os.environ.get("FMP_API_KEY", "")
        self.apify_token = os.environ.get("APIFY_TOKEN", "")
        self.base_url = "https://financialmodelingprep.com/api/v3"
    
    def fetch_transcript(self, ticker: str, year: int, quarter: int) -> Dict:
        quarter_map = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
        quarter_str = quarter_map.get(quarter, f'Q{quarter}')
        
        print(f"Fetching {ticker} {quarter_str} {year}")
        
        # Try FMP first 
        if self.fmp_api_key:
            result = self._fetch_from_fmp(ticker, year, quarter_str)
            if result and result.get('status') == 'success':
                return result
        
        
        if self.apify_token:
            result = self._fetch_from_apify(ticker, year, quarter)
            if result and result.get('status') == 'success':
                return result
        
        # Both sources failed
        return {
            'status': 'error',
            'error_code': 'NO_SOURCE_AVAILABLE',
            'error_message': f'No transcript source available for {ticker} {quarter_str} {year}. Set FMP_API_KEY or APIFY_TOKEN environment variable.',
            'timestamp': datetime.now().isoformat()
        }
    
    def _fetch_from_fmp(self, ticker: str, year: int, quarter_str: str) -> Dict:
        url = f"{self.base_url}/earning_call_transcript"
        params = {
            'symbol': ticker,
            'quarter': quarter_str,
            'year': year,
            'apikey': self.fmp_api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            print(f"FMP response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    content = data[0].get('content', '')
                    if content and len(content) > 200:
                        return {
                            'status': 'success',
                            'source': 'Financial Modeling Prep',
                            'content': content,
                            'source_used': 'FMP API',
                            'timestamp': datetime.now().isoformat()
                        }
        except Exception as e:
            print(f"FMP error: {str(e)}")
        
        return None
    
    def _fetch_from_apify(self, ticker: str, year: int, quarter: int) -> Dict:
        try:
            from apify_client import ApifyClient
        except ImportError:
            print("Apify client not installed. Run: pip install apify-client")
            return None
        
        quarter_map = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
        quarter_str = quarter_map.get(quarter, f'Q{quarter}')
        
        client = ApifyClient(self.apify_token)
        
        run_input = {
            "companySymbol": ticker,
            "quarter": quarter_str,
            "year": year,
        }
        
        try:
            print(f"Calling Apify for {ticker} {quarter_str} {year}")
            run = client.actor("junipr/earnings-call-scraper").call(run_input=run_input)
            
            if run and run.get('status') == 'SUCCEEDED':
                dataset_client = client.dataset(run.get('defaultDatasetId'))
                items = list(dataset_client.iterate_items())
                
                if items and len(items) > 0:
                    content = items[0].get('transcript', '') or items[0].get('content', '')
                    if content and len(content) > 200:
                        return {
                            'status': 'success',
                            'source': 'Apify',
                            'content': content,
                            'source_used': 'Apify Earnings Call Scraper',
                            'timestamp': datetime.now().isoformat()
                        }
        except Exception as e:
            print(f"Apify error: {str(e)}")
        
        return None
