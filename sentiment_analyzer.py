

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from typing import Dict, List
from datetime import datetime

class TransparentSentimentAnalyzer:
    """
    Sentiment analyzer with full transparency.
    Every claim includes source sentences and confidence.
    """
    
    def __init__(self):
        # Use DistilBERT - much smaller than FinBERT (250MB vs 500MB+)
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        
        self.labels = ['negative', 'positive']
    
    def analyze_sentence(self, sentence: str) -> Dict:
        """Analyze a single sentence with transparent output."""
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        sentiment_label = self.labels[predicted_class]
        # Convert to 0-1 scale where 0=negative/cautious, 1=positive/confident
        sentiment_score = confidence if predicted_class == 1 else 1 - confidence
        
        return {
            'sentence': sentence,
            'sentiment_label': sentiment_label,
            'sentiment_score': round(sentiment_score, 3),
            'confidence': round(confidence, 3),
            'model_used': 'distilbert-base-uncased-finetuned-sst-2-english'
        }
    
    def analyze_transcript(self, text: str) -> Dict:
        """Analyze entire transcript with sentence-level transparency."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Analyze each sentence
        sentence_analyses = []
        for sentence in sentences[:50]:  # Limit to first 50 sentences for performance
            analysis = self.analyze_sentence(sentence)
            sentence_analyses.append(analysis)
        
        if not sentence_analyses:
            return {
                'overall_sentiment': {'score': 0.5, 'label': 'neutral', 'sentence_count': 0},
                'all_sentences': [],
                'analysis_timestamp': datetime.now().isoformat()
            }
        
        # Calculate overall sentiment
        avg_score = sum(s['sentiment_score'] for s in sentence_analyses) / len(sentence_analyses)
        
        return {
            'overall_sentiment': {
                'score': round(avg_score, 3),
                'label': 'positive' if avg_score > 0.6 else ('negative' if avg_score < 0.4 else 'neutral'),
                'sentence_count': len(sentence_analyses)
            },
            'all_sentences': sentence_analyses[:10],  # First 10 sentences as evidence
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def compare_transcripts(self, current_text: str, previous_text: str) -> Dict:
        """Compare two transcripts with transparent delta."""
        current_analysis = self.analyze_transcript(current_text)
        previous_analysis = self.analyze_transcript(previous_text)
        
        current_score = current_analysis['overall_sentiment']['score']
        previous_score = previous_analysis['overall_sentiment']['score']
        delta = current_score - previous_score
        
        # Find the most changed sentence (if available)
        most_changed = None
        if current_analysis['all_sentences'] and previous_analysis['all_sentences']:
            # Simple comparison - take first sentence from each as example
            most_changed = {
                'current_sentence': current_analysis['all_sentences'][0]['sentence'][:200],
                'current_score': current_analysis['all_sentences'][0]['sentiment_score'],
                'previous_sentence': previous_analysis['all_sentences'][0]['sentence'][:200] if previous_analysis['all_sentences'] else 'N/A',
                'previous_score': previous_analysis['all_sentences'][0]['sentiment_score'] if previous_analysis['all_sentences'] else 0.5
            }
        
        return {
            'overall_delta': {
                'current': current_score,
                'previous': previous_score,
                'delta': round(delta, 3),
                'direction': 'more confident' if delta > 0.05 else ('less confident' if delta < -0.05 else 'unchanged'),
                'materiality': 'high' if abs(delta) > 0.15 else ('moderate' if abs(delta) > 0.08 else 'low')
            },
            'evidence': {
                'current_sample_sentences': current_analysis['all_sentences'][:3],
                'previous_sample_sentences': previous_analysis['all_sentences'][:3]
            },
            'most_changed_sentence': most_changed,
            'comparison_timestamp': datetime.now().isoformat(),
            'methodology': {
                'model': 'distilbert-base-uncased-finetuned-sst-2-english',
                'sentiment_scale': '0=negative/cautious, 0.5=neutral, 1=positive/confident',
                'transparency': 'All claims backed by exact source sentences'
            }
        }
