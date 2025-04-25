import requests
import os
import time
import json
from typing import List, Dict, Any, Optional

class WebSearchTool:
    """
    Tool for performing web searches and retrieving search results.
    
    This tool can use either a real search API (like SerpAPI) or a mock implementation.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_mock: bool = False):
        """
        Initialize the WebSearchTool.
        
        Args:
            api_key: API key for the search service (if using a real service)
            use_mock: Whether to use a mock implementation instead of a real API
        """
        self.api_key = api_key or os.environ.get("SEARCH_API_KEY")
        self.use_mock = use_mock
        self.base_url = "https://serpapi.com/search"
        
    def search(self, query: str, num_results: int = 10, search_type: str = "web") -> Dict[str, Any]:
        """
        Perform a web search with the given query.
        
        Args:
            query: The search query
            num_results: Number of results to retrieve
            search_type: Type of search (web, news, etc.)
            
        Returns:
            Dictionary containing search results
        """
        if self.use_mock:
            return self._mock_search(query, num_results, search_type)
        
        params = {
            "q": query,
            "num": num_results,
            "api_key": self.api_key,
        }
        
        if search_type == "news":
            params["tbm"] = "nws"
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Search error: {e}")
            # Fall back to mock if real search fails
            return self._mock_search(query, num_results, search_type)
    
    def _mock_search(self, query: str, num_results: int, search_type: str) -> Dict[str, Any]:
        """
        Mock implementation of search for testing or when API is unavailable.
        
        Args:
            query: The search query
            num_results: Number of results to retrieve
            search_type: Type of search (web, news, etc.)
            
        Returns:
            Dictionary containing mock search results
        """
        # Create mock search results based on the query
        results = []
        for i in range(min(num_results, 5)):  # Limit mock results to 5
            results.append({
                "title": f"Result {i+1} for {query}",
                "link": f"https://example.com/result{i+1}",
                "snippet": f"This is a mock search result {i+1} for the query '{query}'.",
                "position": i+1
            })
        
        return {
            "search_metadata": {
                "status": "Success",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_time_taken": 0.5,
                "query": query
            },
            "search_parameters": {
                "q": query,
                "num": num_results
            },
            "organic_results": results
        }
    
    def get_next_page(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the next page of search results.
        
        Args:
            previous_results: Previous search results containing pagination info
            
        Returns:
            Dictionary containing the next page of search results
        """
        if self.use_mock:
            # For mock, just return fewer results to simulate end of results
            query = previous_results["search_parameters"]["q"]
            num_results = max(1, previous_results["search_parameters"]["num"] - 3)
            return self._mock_search(query, num_results, "web")
        
        # In a real implementation, we would extract the next page token/URL
        # from previous_results and make a new request
        
        # For now, just return an empty result to indicate no more pages
        return {"organic_results": []}
