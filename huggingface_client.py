"""
Hugging Face Inference API Client
Uses free tier - no local ML models needed.
"""

import requests
import os
from typing import Dict, List

class HuggingFaceClient:
    def __init__(self):
        # Get token from environment (optional, free tier works without it)
        self.api_token = os.environ.get("HF_TOKEN", "")
        self.headers = {
            "Authorization": f"Bearer {self.api_token}" if self.api_token else "",
            "Content-Type": "application/json"
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using Hugging Face's free inference API.
        Uses distilbert-base-uncased-finetuned-sst-2-english
        """
        # Truncate to 500 chars to keep API call fast
        truncated_text = text[:500]
        
        api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
        
        try:
            response = requests.post(
                api_url,
                headers=self.headers,
                json={"inputs": truncated_text},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    # Find positive and negative scores
                    scores = {item['label']: item['score'] for item in result[0]}
                    positive_score = scores.get('POSITIVE', 0.5)
                    negative_score = scores.get('NEGATIVE', 0.5)
                    
                    # Convert to 0-1 scale (0=negative, 1=positive)
                    sentiment_score = positive_score
                    
                    return {
                        'sentiment_label': 'positive' if positive_score > 0.6 else ('negative' if negative_score > 0.6 else 'neutral'),
                        'sentiment_score': round(sentiment_score, 3),
                        'confidence': round(max(positive_score, negative_score), 3),
                        'model_used': 'distilbert-base-uncased-finetuned-sst-2-english',
                        'api': 'HuggingFace Inference (free)'
                    }
            
            # Fallback
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5,
                'model_used': 'fallback',
                'note': 'Using fallback due to API error'
            }
            
        except Exception as e:
            print(f"Sentiment API error: {e}")
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5,
                'error': str(e)[:100]
            }
