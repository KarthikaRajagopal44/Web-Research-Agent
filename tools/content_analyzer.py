from typing import Dict, Any, List, Optional, Tuple
import re
import datetime
from collections import Counter

class ContentAnalyzer:
    """
    Tool for analyzing and processing extracted web content.
    
    This tool evaluates content for relevance, reliability, and information value.
    """
    
    def __init__(self, ai_model=None):
        """
        Initialize the ContentAnalyzer.
        
        Args:
            ai_model: AI model to use for advanced analysis (optional)
        """
        self.ai_model = ai_model
        
    def analyze(self, content: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Analyze the content for relevance to the query.
        
        Args:
            content: Extracted content from a web page
            query: Original search query
            
        Returns:
            Dictionary containing analysis results
        """
        # Basic analysis without AI model
        relevance_score = self._calculate_relevance(content, query)
        information_density = self._calculate_information_density(content)
        recency = self._estimate_recency(content)
        reliability = self._estimate_reliability(content)
        
        # Extract key sentences related to the query
        key_sentences = self._extract_key_sentences(content, query)
        
        # Use AI model for advanced analysis if available
        ai_analysis = {}
        if self.ai_model:
            ai_analysis = self._perform_ai_analysis(content, query)
        
        return {
            "relevance_score": relevance_score,
            "information_density": information_density,
            "recency": recency,
            "reliability": reliability,
            "key_sentences": key_sentences,
            "ai_analysis": ai_analysis,
        }
    
    def _calculate_relevance(self, content: Dict[str, Any], query: str) -> float:
        """
        Calculate the relevance of the content to the query.
        
        Args:
            content: Extracted content
            query: Search query
            
        Returns:
            Relevance score between 0 and 1
        """
        # Simple relevance calculation based on term frequency
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        # Get text content
        text = content.get("main_content", "") or content.get("full_text", "")
        text = text.lower()
        
        # Count query terms in the content
        term_count = 0
        for term in query_terms:
            term_count += len(re.findall(r'\b' + re.escape(term) + r'\b', text))
        
        # Calculate relevance score
        if not query_terms:
            return 0.0
        
        # Check if query terms appear in title (higher weight)
        title = content.get("title", "").lower()
        title_matches = sum(1 for term in query_terms if term in title)
        
        # Check if query terms appear in headings (medium weight)
        headings_text = " ".join([h["text"].lower() for h in content.get("headings", [])])
        heading_matches = sum(1 for term in query_terms if term in headings_text)
        
        # Combine scores with weights
        score = (term_count / len(query_terms)) * 0.6 + (title_matches / len(query_terms)) * 0.3 + (heading_matches / len(query_terms)) * 0.1
        
        return min(1.0, score)
    
    def _calculate_information_density(self, content: Dict[str, Any]) -> float:
        """
        Calculate the information density of the content.
        
        Args:
            content: Extracted content
            
        Returns:
            Information density score between 0 and 1
        """
        text = content.get("main_content", "") or content.get("full_text", "")
        
        if not text:
            return 0.0
        
        # Count structural elements that indicate information
        tables_count = len(content.get("tables", []))
        lists_count = len(content.get("lists", []))
        headings_count = len(content.get("headings", []))
        
        # Count sentences
        sentences = re.split(r'[.!?]+', text)
        sentence_count = sum(1 for s in sentences if len(s.strip()) > 0)
        
        if sentence_count == 0:
            return 0.0
        
        # Calculate average sentence length (words)
        words = re.findall(r'\b\w+\b', text)
        avg_sentence_length = len(words) / sentence_count if sentence_count > 0 else 0
        
        # Calculate unique word ratio (vocabulary richness)
        unique_words = len(set(words))
        unique_word_ratio = unique_words / len(words) if words else 0
        
        # Calculate structural element density
        structural_density = min(1.0, (tables_count + lists_count + headings_count) / 10)
        
        # Combine metrics
        density = (
            0.4 * unique_word_ratio + 
            0.3 * min(1.0, avg_sentence_length / 20) +  # Cap at 20 words per sentence
            0.3 * structural_density
        )
        
        return density
    
    def _estimate_recency(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate how recent the content is.
        
        Args:
            content: Extracted content
            
        Returns:
            Dictionary with recency information
        """
        # Check metadata for publication date
        metadata = content.get("metadata", {})
        date_fields = ["article:published_time", "datePublished", "date", "pubdate", "og:published_time"]
        
        for field in date_fields:
            if field in metadata:
                try:
                    date_str = metadata[field]
                    # Try to parse the date
                    date = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    return {
                        "found": True,
                        "date": date.isoformat(),
                        "days_ago": (datetime.datetime.now(datetime.timezone.utc) - date).days,
                        "source": field
                    }
                except (ValueError, TypeError):
                    pass
        
        # Look for dates in the text
        text = content.get("main_content", "") or content.get("full_text", "")
        
        # Simple date pattern matching (can be improved)
        date_patterns = [
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return {
                    "found": True,
                    "date_text": matches[0],
                    "source": "text_pattern"
                }
        
        # If no date found
        return {
            "found": False,
            "source": None
        }
    
    def _estimate_reliability(self, content: Dict[str, Any]) -> float:
        """
        Estimate the reliability of the content.
        
        Args:
            content: Extracted content
            
        Returns:
            Reliability score between 0 and 1
        """
        # This is a simplified reliability estimation
        # In a real implementation, this would include:
        # - Domain reputation checking
        # - Author credentials
        # - Citation and reference analysis
        # - Fact checking against known reliable sources
        
        text = content.get("main_content", "") or content.get("full_text", "")
        
        if not text:
            return 0.0
        
        # Check for citations and references
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'$$\d{4}$$',  # (2020), (2021), etc.
            r'et al\.',  # et al.
            r'according to',  # Attribution phrases
            r'cited by',
            r'referenced in'
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, text))
        
        # Check for balanced reporting (presence of qualifying language)
        qualifying_terms = [
            r'\bmay\b', r'\bcould\b', r'\bpossibly\b', r'\bpotentially\b',
            r'\bsuggests\b', r'\bindicates\b', r'\bappears\b'
        ]
        
        qualifying_count = 0
        for term in qualifying_terms:
            qualifying_count += len(re.findall(term, text))
        
        # Calculate basic reliability score
        word_count = len(re.findall(r'\b\w+\b', text))
        
        if word_count < 100:
            # Very short content is less reliable
            base_score = 0.3
        else:
            base_score = 0.5
        
        # Adjust score based on citations and qualifying language
        citation_score = min(0.3, citation_count * 0.03)
        qualifying_score = min(0.2, qualifying_count * 0.02)
        
        reliability = base_score + citation_score + qualifying_score
        
        return min(1.0, reliability)
    
    def _extract_key_sentences(self, content: Dict[str, Any], query: str) -> List[str]:
        """
        Extract key sentences related to the query.
        
        Args:
            content: Extracted content
            query: Search query
            
        Returns:
            List of key sentences
        """
        text = content.get("main_content", "") or content.get("full_text", "")
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score sentences based on query term presence
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
                
            score = 0
            for term in query_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', sentence.lower()):
                    score += 1
            
            if score > 0:
                scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:5]]
        
        return top_sentences
    
    def _perform_ai_analysis(self, content: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Perform advanced analysis using AI model.
        
        Args:
            content: Extracted content
            query: Search query
            
        Returns:
            Dictionary with AI analysis results
        """
        if not self.ai_model:
            return {}
        
        text = content.get("main_content", "") or content.get("full_text", "")
        
        # Truncate text if too long
        max_length = 4000  # Adjust based on model context limits
        if len(text) > max_length:
            text = text[:max_length]
        
        try:
            # This is a placeholder for actual AI model integration
            # In a real implementation, this would call the AI model API
            
            # Example prompt for the AI model
            prompt = f"""
            Analyze the following content in relation to the query: "{query}"
            
            Content:
            {text}
            
            Please provide:
            1. A relevance assessment (0-10)
            2. Key facts related to the query
            3. Any potential biases or limitations in the content
            4. Confidence in the information (0-10)
            """
            
            # Mock AI response for demonstration
            ai_response = {
                "relevance_assessment": 7,
                "key_facts": [
                    "This is a mock key fact 1",
                    "This is a mock key fact 2",
                    "This is a mock key fact 3"
                ],
                "potential_biases": "This is a mock bias analysis",
                "confidence": 8
            }
            
            return ai_response
        except Exception as e:
            print(f"AI analysis error: {e}")
            return {}
