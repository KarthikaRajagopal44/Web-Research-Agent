import requests
import time
from typing import List, Dict, Any, Optional
import datetime
import re
from urllib.parse import urlencode

class NewsAggregator:
    """
    Tool for finding and filtering recent news articles on specific topics.
    
    This tool can use either a real news API or a mock implementation.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_mock: bool = False):
        """
        Initialize the NewsAggregator.
        
        Args:
            api_key: API key for the news service (if using a real service)
            use_mock: Whether to use a mock implementation instead of a real API
        """
        self.api_key = api_key
        self.use_mock = use_mock
        self.base_url = "https://newsapi.org/v2/everything"
    
    def search_news(self, query: str, days: int = 7, language: str = "en", sort_by: str = "relevancy", max_results: int = 10) -> Dict[str, Any]:
        """
        Search for news articles related to the query.
        
        Args:
            query: Search query
            days: Number of days to look back
            language: Language of the articles
            sort_by: Sorting method (relevancy, popularity, publishedAt)
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing news search results
        """
        if self.use_mock:
            return self._mock_search_news(query, days, language, sort_by, max_results)
        
        # Calculate date range
        from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        
        params = {
            "q": query,
            "from": from_date,
            "language": language,
            "sortBy": sort_by,
            "pageSize": max_results,
            "apiKey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"News search error: {e}")
            # Fall back to mock if real search fails
            return self._mock_search_news(query, days, language, sort_by, max_results)
    
    def _mock_search_news(self, query: str, days: int, language: str, sort_by: str, max_results: int) -> Dict[str, Any]:
        """
        Mock implementation of news search for testing or when API is unavailable.
        
        Args:
            query: Search query
            days: Number of days to look back
            language: Language of the articles
            sort_by: Sorting method
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing mock news search results
        """
        # Create mock news results based on the query
        articles = []
        for i in range(min(max_results, 5)):  # Limit mock results to 5
            # Generate a random date within the specified range
            days_ago = int(i * days / 5)
            article_date = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).isoformat()
            
            articles.append({
                "source": {
                    "id": f"mock-source-{i}",
                    "name": f"Mock News Source {i}"
                },
                "author": f"Author {i}",
                "title": f"News Article {i} about {query}",
                "description": f"This is a mock news article {i} about {query}.",
                "url": f"https://example.com/news/{i}",
                "urlToImage": f"https://example.com/images/news{i}.jpg",
                "publishedAt": article_date,
                "content": f"This is the content of mock news article {i} about {query}. It contains information that would be relevant to the search query."
            })
        
        return {
            "status": "ok",
            "totalResults": len(articles),
            "articles": articles
        }
    
    def filter_news(self, news_results: Dict[str, Any], keywords: List[str] = None, exclude_keywords: List[str] = None, min_length: int = 0) -> Dict[str, Any]:
        """
        Filter news results based on criteria.
        
        Args:
            news_results: News search results
            keywords: Keywords that must be present in the article
            exclude_keywords: Keywords that must not be present in the article
            min_length: Minimum content length
            
        Returns:
            Dictionary containing filtered news results
        """
        if "articles" not in news_results:
            return news_results
        
        filtered_articles = []
        
        for article in news_results["articles"]:
            # Combine title, description, and content for filtering
            article_text = " ".join([
                article.get("title", ""),
                article.get("description", ""),
                article.get("content", "")
            ]).lower()
            
            # Check if all required keywords are present
            if keywords and not all(keyword.lower() in article_text for keyword in keywords):
                continue
            
            # Check if any excluded keywords are present
            if exclude_keywords and any(keyword.lower() in article_text for keyword in exclude_keywords):
                continue
            
            # Check minimum content length
            if min_length > 0 and len(article_text) < min_length:
                continue
            
            filtered_articles.append(article)
        
        return {
            "status": news_results.get("status", "ok"),
            "totalResults": len(filtered_articles),
            "articles": filtered_articles
        }
    
    def get_trending_topics(self, category: str = "general", country: str = "us") -> List[str]:
        """
        Get trending news topics.
        
        Args:
            category: News category
            country: Country code
            
        Returns:
            List of trending topics
        """
        if self.use_mock:
            return self._mock_trending_topics(category)
        
        # In a real implementation, this would call a trending topics API
        # For now, return mock trending topics
        return self._mock_trending_topics(category)
    
    def _mock_trending_topics(self, category: str) -> List[str]:
        """
        Mock implementation of trending topics.
        
        Args:
            category: News category
            
        Returns:
            List of mock trending topics
        """
        topics_by_category = {
            "technology": ["Artificial Intelligence", "Quantum Computing", "Cybersecurity", "5G Networks", "Blockchain"],
            "business": ["Global Economy", "Stock Market", "Startup Funding", "Corporate Mergers", "Supply Chain"],
            "health": ["Pandemic Response", "Medical Research", "Healthcare Policy", "Mental Health", "Vaccines"],
            "science": ["Space Exploration", "Climate Change", "Genetic Engineering", "Renewable Energy", "Neuroscience"],
            "sports": ["Olympic Games", "Football Championships", "Tennis Tournaments", "Basketball Leagues", "Athletics"],
            "entertainment": ["Film Festivals", "Music Awards", "Celebrity News", "Streaming Services", "Gaming Industry"],
            "general": ["International Relations", "Political Developments", "Environmental Issues", "Education Reform", "Social Movements"]
        }
        
        return topics_by_category.get(category.lower(), topics_by_category["general"])
