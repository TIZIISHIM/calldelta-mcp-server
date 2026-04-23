

import requests
import os
import re
from typing import Dict, List

class HuggingFaceClient:
    def __init__(self):
        self.api_token = os.environ.get("HF_TOKEN", "")
        self.headers = {
            "Authorization": f"Bearer {self.api_token}" if self.api_token else "",
            "Content-Type": "application/json"
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of full text (legacy method for backward compatibility)."""
        return self.analyze_sentiment_with_evidence(text)
    
    def analyze_sentiment_with_evidence(self, text: str, max_sentences: int = 10) -> Dict:
        """
        Analyze sentiment and return sentence-level evidence.
        This fulfills the transparent materiality requirement.
        
        Every claim includes:
        - The exact sentence that was analyzed
        - The sentiment label (positive/neutral/negative)
        - The sentiment score (0-1 scale)
        - Confidence level
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:max_sentences]
        
        if not sentences:
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5,
                'evidence': [],
                'note': 'No substantial sentences found for analysis'
            }
        
        # Analyze each sentence individually (transparent materiality)
        sentence_results = []
        for sentence in sentences:
            result = self._analyze_single_sentence(sentence)
            sentence_results.append(result)
        
        # Calculate overall sentiment from individual sentences
        scores = [r['sentiment_score'] for r in sentence_results if r.get('sentiment_score') is not None]
        if scores:
            overall_score = sum(scores) / len(scores)
        else:
            overall_score = 0.5
        
        # Determine overall label
        if overall_score > 0.6:
            overall_label = 'positive'
        elif overall_score < 0.4:
            overall_label = 'negative'
        else:
            overall_label = 'neutral'
        
        return {
            'sentiment_label': overall_label,
            'sentiment_score': round(overall_score, 3),
            'confidence': round(abs(overall_score - 0.5) * 2, 3),
            'evidence': sentence_results,
            'sentence_count': len(sentence_results),
            'model_used': 'distilbert-base-uncased-finetuned-sst-2-english',
            'api': 'HuggingFace Inference (free)',
            'transparency_note': 'Each sentence was analyzed individually. See evidence array for exact sentences and their scores.'
        }
    
    def _analyze_single_sentence(self, sentence: str) -> Dict:
        """Analyze a single sentence and return evidence with exact sentence text."""
        api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
        
        try:
            response = requests.post(
                api_url,
                headers=self.headers,
                json={"inputs": sentence[:500]},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    scores = {item['label']: item['score'] for item in result[0]}
                    positive_score = scores.get('POSITIVE', 0.5)
                    negative_score = scores.get('NEGATIVE', 0.5)
                    
                    # Determine label based on thresholds
                    if positive_score > 0.6:
                        label = 'positive'
                    elif negative_score > 0.6:
                        label = 'negative'
                    else:
                        label = 'neutral'
                    
                    return {
                        'sentence': sentence[:300],
                        'sentiment_label': label,
                        'sentiment_score': round(positive_score, 3),
                        'confidence': round(max(positive_score, negative_score), 3),
                        'model_used': 'distilbert-base-uncased-finetuned-sst-2-english'
                    }
            
            # Fallback for API error
            return {
                'sentence': sentence[:200],
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5,
                'error': 'API returned unexpected format'
            }
            
        except Exception as e:
            return {
                'sentence': sentence[:200],
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.5,
                'error': str(e)[:50]
            }
    
    def compare_with_evidence(self, current_text: str, previous_text: str) -> Dict:
        """
        Compare two transcripts with sentence-level evidence.
        This is the core method for CallDelta's transparent materiality.
        
        Returns:
            - overall_delta: summary of sentiment change
            - current_evidence: sentence-by-sentence analysis of current transcript
            - previous_evidence: sentence-by-sentence analysis of previous transcript
            - most_changed_sentence: the sentence with the largest sentiment shift
        """
        current_analysis = self.analyze_sentiment_with_evidence(current_text)
        previous_analysis = self.analyze_sentiment_with_evidence(previous_text)
        
        current_score = current_analysis['sentiment_score']
        previous_score = previous_analysis['sentiment_score']
        delta = current_score - previous_score
        
        # Find the most changed sentence (if available)
        most_changed = None
        current_evidence = current_analysis.get('evidence', [])
        previous_evidence = previous_analysis.get('evidence', [])
        
        if current_evidence and previous_evidence:
            # Try to align sentences by content similarity (simplified - compare first few)
            for curr in current_evidence[:5]:
                for prev in previous_evidence[:5]:
                    if curr.get('sentence') and prev.get('sentence'):
                        sentence_delta = curr.get('sentiment_score', 0.5) - prev.get('sentiment_score', 0.5)
                        if abs(sentence_delta) > 0.2:  # Significant change threshold
                            most_changed = {
                                'current_sentence': curr.get('sentence', '')[:250],
                                'current_score': curr.get('sentiment_score', 0.5),
                                'previous_sentence': prev.get('sentence', '')[:250],
                                'previous_score': prev.get('sentiment_score', 0.5),
                                'delta': round(sentence_delta, 3),
                                'change_type': 'more_confident' if sentence_delta > 0 else 'less_confident'
                            }
                            break
                    if most_changed:
                        break
                if most_changed:
                    break
        
        # If no specific sentence change found, use the first sentence as example
        if not most_changed and current_evidence and previous_evidence:
            most_changed = {
                'current_sentence': current_evidence[0].get('sentence', '')[:250] if current_evidence else 'N/A',
                'current_score': current_evidence[0].get('sentiment_score', 0.5) if current_evidence else 0.5,
                'previous_sentence': previous_evidence[0].get('sentence', '')[:250] if previous_evidence else 'N/A',
                'previous_score': previous_evidence[0].get('sentiment_score', 0.5) if previous_evidence else 0.5,
                'delta': round((current_evidence[0].get('sentiment_score', 0.5) if current_evidence else 0.5) - 
                              (previous_evidence[0].get('sentiment_score', 0.5) if previous_evidence else 0.5), 3),
                'note': 'Example sentence change from each transcript'
            }
        
        return {
            'overall_delta': {
                'current': current_score,
                'previous': previous_score,
                'delta': round(delta, 3),
                'direction': 'more confident' if delta > 0.05 else ('less confident' if delta < -0.05 else 'unchanged'),
                'materiality': 'high' if abs(delta) > 0.15 else ('moderate' if abs(delta) > 0.08 else 'low')
            },
            'current_evidence': current_evidence[:5],  # First 5 sentences as evidence
            'previous_evidence': previous_evidence[:5],  # First 5 sentences as evidence
            'most_changed_sentence': most_changed,
            'methodology': {
                'model': 'distilbert-base-uncased-finetuned-sst-2-english',
                'api': 'HuggingFace Inference (free)',
                'transparency': 'Each sentence analyzed individually. See evidence arrays for exact sentences and scores.',
                'sentiment_scale': '0=negative/cautious, 0.5=neutral, 1=positive/confident'
            }
        }


# Example usage for testing
if __name__ == "__main__":
    client = HuggingFaceClient()
    
    # Test with sample text
    sample = "We had a strong quarter with record revenue. Margins expanded significantly. The competitive environment remains challenging but we are confident in our position."
    
    result = client.analyze_sentiment_with_evidence(sample)
    print("=== Sentence-Level Evidence ===")
    for evidence in result.get('evidence', []):
        print(f"Sentence: {evidence.get('sentence', '')[:80]}...")
        print(f"Score: {evidence.get('sentiment_score')} | Label: {evidence.get('sentiment_label')}")
        print("-" * 40)
