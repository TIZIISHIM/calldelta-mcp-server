

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from typing import Dict, List, Tuple
from datetime import datetime

class TransparentSentimentAnalyzer:
    """
    Sentiment analyzer that provides full transparency.
    No black-box verdicts. Every claim includes source sentences and confidence.
    """
    
    def __init__(self):
        # Load FinBERT model
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        
        self.labels = ['negative', 'neutral', 'positive']
        
        # Topic keywords for section detection
        self.topic_keywords = {
            'revenue': ['revenue', 'sales', 'top line', 'income', 'billings'],
            'guidance': ['guidance', 'outlook', 'expect', 'forecast', 'project', 'expectation'],
            'margin': ['margin', 'gross margin', 'operating margin', 'profitability', 'profit'],
            'demand': ['demand', 'customer', 'orders', 'pipeline', 'backlog', 'bookings'],
            'competitive': ['competitive', 'competition', 'market share', 'rival', 'landscape'],
            'cost': ['cost', 'expense', 'spending', 'inflation', 'supply chain', 'opex']
        }
    
    def analyze_sentence(self, sentence: str) -> Dict:
        """
        Analyze a single sentence and return transparent results.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        sentiment_label = self.labels[predicted_class]
        sentiment_score = self._to_confidence_score(predicted_class, confidence)
        
        return {
            'sentence': sentence,
            'sentiment_label': sentiment_label,
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'model_used': 'FinBERT',
            'model_version': 'ProsusAI/finbert'
        }
    
    def _to_confidence_score(self, predicted_class: int, confidence: float) -> float:
        """Convert FinBERT output to 0-1 scale."""
        if predicted_class == 0:  # negative
            return confidence * 0.33
        elif predicted_class == 1:  # neutral
            return 0.33 + (confidence * 0.33)
        else:  # positive
            return 0.66 + (confidence * 0.34)
    
    def analyze_transcript(self, text: str) -> Dict:
        """
        Analyze entire transcript and return structured results with transparency.
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # Analyze each sentence
        sentence_analyses = []
        for sentence in sentences:
            if len(sentence) > 20:  # Skip very short sentences
                analysis = self.analyze_sentence(sentence)
                sentence_analyses.append(analysis)
        
        # Group by topic
        topic_analyses = {}
        for topic, keywords in self.topic_keywords.items():
            topic_sentences = []
            for analysis in sentence_analyses:
                sentence_lower = analysis['sentence'].lower()
                if any(keyword in sentence_lower for keyword in keywords):
                    topic_sentences.append(analysis)
            
            if topic_sentences:
                avg_score = sum(s['sentiment_score'] for s in topic_sentences) / len(topic_sentences)
                topic_analyses[topic] = {
                    'topic': topic,
                    'average_sentiment_score': round(avg_score, 3),
                    'sentiment_label': self._score_to_label(avg_score),
                    'sentence_count': len(topic_sentences),
                    'evidence_sentences': topic_sentences[:5]  # Top 5 sentences as evidence
                }
        
        # Calculate overall sentiment
        overall_score = sum(s['sentiment_score'] for s in sentence_analyses) / len(sentence_analyses) if sentence_analyses else 0.5
        
        return {
            'overall_sentiment': {
                'score': round(overall_score, 3),
                'label': self._score_to_label(overall_score),
                'sentence_count': len(sentence_analyses)
            },
            'topic_breakdown': topic_analyses,
            'all_sentences': sentence_analyses[:20],  # First 20 sentences for transparency
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def compare_transcripts(self, current_text: str, previous_text: str) -> Dict:
        """
        Compare two transcripts and return transparent delta.
        This is the core method for CallDelta.
        """
        # Analyze both transcripts
        current_analysis = self.analyze_transcript(current_text)
        previous_analysis = self.analyze_transcript(previous_text)
        
        # Calculate overall delta
        current_overall = current_analysis['overall_sentiment']['score']
        previous_overall = previous_analysis['overall_sentiment']['score']
        overall_delta = current_overall - previous_overall
        
        # Calculate topic-level deltas
        topic_deltas = {}
        all_topics = set(current_analysis['topic_breakdown'].keys()) | set(previous_analysis['topic_breakdown'].keys())
        
        for topic in all_topics:
            current_topic = current_analysis['topic_breakdown'].get(topic, {})
            previous_topic = previous_analysis['topic_breakdown'].get(topic, {})
            
            current_score = current_topic.get('average_sentiment_score', 0.5)
            previous_score = previous_topic.get('average_sentiment_score', 0.5)
            delta = current_score - previous_score
            
            topic_deltas[topic] = {
                'topic': topic,
                'current_sentiment': current_score,
                'previous_sentiment': previous_score,
                'delta': round(delta, 3),
                'direction': 'more confident' if delta > 0.05 else ('less confident' if delta < -0.05 else 'unchanged'),
                'evidence': {
                    'current_example_sentences': current_topic.get('evidence_sentences', [])[:3],
                    'previous_example_sentences': previous_topic.get('evidence_sentences', [])[:3]
                }
            }
        
        # Identify the most material change
        material_changes = [t for t in topic_deltas.values() if abs(t['delta']) > 0.1]
        most_material = max(material_changes, key=lambda x: abs(x['delta'])) if material_changes else None
        
        return {
            'overall_delta': {
                'current': current_overall,
                'previous': previous_overall,
                'delta': round(overall_delta, 3),
                'direction': 'more confident' if overall_delta > 0.05 else ('less confident' if overall_delta < -0.05 else 'unchanged'),
                'materiality': 'high' if abs(overall_delta) > 0.15 else ('moderate' if abs(overall_delta) > 0.08 else 'low')
            },
            'topic_deltas': topic_deltas,
            'most_material_change': most_material,
            'current_analysis': current_analysis,
            'previous_analysis': previous_analysis,
            'comparison_timestamp': datetime.now().isoformat(),
            'methodology': {
                'model': 'FinBERT (ProsusAI/finbert)',
                'sentiment_scale': '0=very negative/cautious, 0.5=neutral, 1=very positive/confident',
                'transparency': 'All claims backed by exact source sentences'
            }
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]
    
    def _score_to_label(self, score: float) -> str:
        """Convert score to label."""
        if score > 0.66:
            return 'positive'
        elif score < 0.33:
            return 'negative'
        else:
            return 'neutral'
