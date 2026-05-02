import requests
import os
import re
from typing import Dict, List

class HuggingFaceClient:
    def __init__(self):
        self.hf_token = os.environ.get("HF_TOKEN", "")
        self.replicate_token = os.environ.get("REPLICATE_API_TOKEN", "")
        self.gradio_space_url = os.environ.get("GRADIO_SPACE_URL", "")
        
        self.hf_headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
        # List of HF model URLs to try
        self.hf_model_urls = [
            "https://api-inference.huggingface.co/models/ProsusAI/finbert",
            "https://api-inference.huggingface.co/models/ahmedrachid/FinancialBERT-Sentiment-Analysis",
            "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        ]
        
        # Track which source is currently working
        self.current_source = "huggingface" if self.hf_token else ("replicate" if self.replicate_token else ("gradio" if self.gradio_space_url else "fallback"))
        
        print(f"Initialized HuggingFaceClient with source: {self.current_source}")
    
    def analyze_sentiment_with_evidence(self, text: str, max_sentences: int = 150) -> Dict:
        # Split into sentences - threshold 15 chars
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15][:max_sentences]
        
        if not sentences:
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.0,
                'evidence': [],
                'sentence_count': 0,
                'error': 'No valid sentences found (minimum 15 characters)'
            }
        
        # Process sentences
        sentence_results = []
        for sentence in sentences:
            result = self._analyze_sentence(sentence)
            sentence_results.append(result)
        
        # Calculate overall sentiment
        scores = [r['sentiment_score'] for r in sentence_results if r.get('sentiment_score') is not None]
        overall_score = sum(scores) / len(scores) if scores else 0.5
        label = 'positive' if overall_score > 0.55 else ('negative' if overall_score < 0.45 else 'neutral')
        
        response = {
            'sentiment_label': label,
            'sentiment_score': round(overall_score, 3),
            'confidence': round(abs(overall_score - 0.5) * 2, 3),
            'evidence': sentence_results[:50],
            'sentence_count': len(sentence_results),
            'source': self.current_source
        }
        
        if self.current_source == "fallback":
            response['warning'] = 'Using rule-based fallback. Set HF_TOKEN, REPLICATE_API_TOKEN, or GRADIO_SPACE_URL for better accuracy.'
        
        return response
    
    def _analyze_sentence(self, sentence: str) -> Dict:
        # Layer 1: Try Hugging Face API
        if self.current_source == "huggingface" and self.hf_token:
            for url in self.hf_model_urls:
                result = self._call_hf_api(url, sentence)
                if result:
                    self.current_source = "huggingface"
                    return result
            print("All HF models failed, switching to Replicate")
            self.current_source = "replicate"
        
        # Layer 2: Try Replicate API
        if self.current_source == "replicate" and self.replicate_token:
            result = self._call_replicate_api(sentence)
            if result:
                self.current_source = "replicate"
                return result
            print("Replicate failed, switching to Gradio Space")
            self.current_source = "gradio"
        
        # Layer 3: Try Gradio Space API
        if self.current_source == "gradio" and self.gradio_space_url:
            result = self._call_gradio_api(sentence)
            if result:
                self.current_source = "gradio"
                return result
            print("Gradio Space failed, switching to rule-based fallback")
            self.current_source = "fallback"
        
        # Layer 4: Rule-based fallback
        return self._fallback_sentiment(sentence)
    
    def _call_hf_api(self, api_url: str, sentence: str) -> Dict:
        try:
            response = requests.post(
                api_url,
                headers=self.hf_headers,
                json={"inputs": sentence[:500]},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0:
                    item = result[0] if isinstance(result, list) else result
                    label = item.get('label', 'neutral').lower()
                    score = item.get('score', 0.5)
                    
                    sentiment_score = score if label == 'positive' else (1 - score if label == 'negative' else 0.5)
                    return {
                        'sentence': sentence[:300],
                        'sentiment_label': label,
                        'sentiment_score': round(sentiment_score, 3),
                        'confidence': round(score, 3),
                        'source': 'hf-api'
                    }
            elif response.status_code == 401:
                print(f"HF API 401: Invalid token at {api_url}")
            else:
                print(f"HF API {response.status_code} at {api_url}")
        except Exception as e:
            print(f"HF API error: {str(e)}")
        return None
    
    def _call_replicate_api(self, sentence: str) -> Dict:
        try:
            import replicate
            
            output = replicate.run(
                "nateraw/prosusai-finbert:latest",
                input={"text": sentence[:500]}
            )
            
            if output:
                label = output.get('label', 'neutral').lower()
                score = output.get('score', 0.5)
                sentiment_score = score if label == 'positive' else (1 - score if label == 'negative' else 0.5)
                return {
                    'sentence': sentence[:300],
                    'sentiment_label': label,
                    'sentiment_score': round(sentiment_score, 3),
                    'confidence': round(score, 3),
                    'source': 'replicate-api'
                }
        except ImportError:
            print("Replicate library not installed. Run: pip install replicate")
        except Exception as e:
            print(f"Replicate API error: {str(e)}")
        return None
    
def _call_gradio_api(self, sentence: str) -> Dict:
    try:
        # Try the correct Gradio API endpoint format
        api_url = f"{self.gradio_space_url}/api/predict/sentiment_analysis"
        
        response = requests.post(
            api_url,
            json={"data": [sentence[:500]]},
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result and 'data' in result:
                result_data = result['data']
                if result_data and len(result_data) > 0:
                    # The result might be a JSON string or a direct value
                    sentiment_str = result_data[0]
                    try:
                        sentiment_data = json.loads(sentiment_str)
                    except:
                        # If not JSON, it might be just the label
                        sentiment_data = {"label": sentiment_str, "score": 0.8, "sentiment_score": 0.8}
                    
                    label = sentiment_data.get('label', 'neutral').lower()
                    score = sentiment_data.get('score', 0.5)
                    sentiment_score = sentiment_data.get('sentiment_score', score)
                    return {
                        'sentence': sentence[:300],
                        'sentiment_label': label,
                        'sentiment_score': round(sentiment_score, 3),
                        'confidence': round(score, 3),
                        'source': 'gradio-space'
                    }
    except Exception as e:
        print(f"Gradio API error: {str(e)}")
    
    # Try alternative endpoint format
    try:
        api_url = f"{self.gradio_space_url}/gradio_api/predict/sentiment_analysis"
        response = requests.post(
            api_url,
            json={"data": [sentence[:500]]},
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result and 'data' in result:
                result_data = result['data']
                if result_data and len(result_data) > 0:
                    sentiment_str = result_data[0]
                    try:
                        sentiment_data = json.loads(sentiment_str)
                    except:
                        sentiment_data = {"label": sentiment_str, "score": 0.8, "sentiment_score": 0.8}
                    
                    label = sentiment_data.get('label', 'neutral').lower()
                    score = sentiment_data.get('score', 0.5)
                    sentiment_score = sentiment_data.get('sentiment_score', score)
                    return {
                        'sentence': sentence[:300],
                        'sentiment_label': label,
                        'sentiment_score': round(sentiment_score, 3),
                        'confidence': round(score, 3),
                        'source': 'gradio-space'
                    }
    except Exception as e:
        print(f"Gradio API alternative error: {str(e)}")
    
    return None
    
    def _fallback_sentiment(self, sentence: str) -> Dict:
        """Rule-based fallback for when all APIs fail."""
        sentence_lower = sentence.lower()
        
        positive_words = [
            'growth', 'grew', 'increase', 'increased', 'rising', 'up', 'higher',
            'record', 'strong', 'confidence', 'confident', 'optimistic', 'beat',
            'exceed', 'outperform', 'raise', 'raised', 'guidance', 'momentum',
            'accelerate', 'accelerated', 'expansion', 'expand', 'profit', 'profitable',
            'outlook', 'improve', 'improved', 'improvement', 'better', 'best'
        ]
        
        negative_words = [
            'decline', 'declined', 'decrease', 'decreased', 'fall', 'fell', 'down', 'lower',
            'weak', 'weakness', 'challenge', 'headwind', 'pressure', 'stress', 'risk',
            'miss', 'missed', 'below', 'reduce', 'reduced', 'cut', 'slash', 'drop', 'dropped',
            'collapsed', 'collapse', 'loss', 'lose', 'uncertain', 'uncertainty',
            'deteriorate', 'deterioration', 'worse', 'worsening', 'problem', 'collapsing'
        ]
        
        pos_score = sum(1 for word in positive_words if word in sentence_lower)
        neg_score = sum(1 for word in negative_words if word in sentence_lower)
        
        if pos_score > neg_score:
            sentiment_score = min(0.6 + (pos_score * 0.05), 0.95)
            label = 'positive'
        elif neg_score > pos_score:
            sentiment_score = max(0.4 - (neg_score * 0.05), 0.05)
            label = 'negative'
        else:
            sentiment_score = 0.5
            label = 'neutral'
        
        confidence = 0.8 if abs(pos_score - neg_score) >= 2 else (0.7 if abs(pos_score - neg_score) >= 1 else 0.5)
        
        return {
            'sentence': sentence[:300],
            'sentiment_label': label,
            'sentiment_score': round(sentiment_score, 3),
            'confidence': round(confidence, 3),
            'source': 'rule-based-fallback'
        }
    
    def compare_with_evidence(self, current_text: str, previous_text: str) -> Dict:
        current = self.analyze_sentiment_with_evidence(current_text)
        previous = self.analyze_sentiment_with_evidence(previous_text)
        
        delta = current['sentiment_score'] - previous['sentiment_score']
        direction = 'more confident' if delta > 0.05 else ('less confident' if delta < -0.05 else 'unchanged')
        materiality = 'high' if abs(delta) > 0.15 else ('moderate' if abs(delta) > 0.08 else 'low')
        
        result = {
            'overall_delta': {
                'current': current['sentiment_score'],
                'previous': previous['sentiment_score'],
                'delta': round(delta, 3),
                'direction': direction,
                'materiality': materiality
            },
            'current_evidence': current.get('evidence', [])[:10],
            'previous_evidence': previous.get('evidence', [])[:10],
            'total_sentences_analyzed': {
                'current': current.get('sentence_count', 0),
                'previous': previous.get('sentence_count', 0)
            }
        }
        
        if current.get('warning'):
            result['warning'] = current['warning']
        
        return result
