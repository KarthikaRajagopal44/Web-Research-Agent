import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
import tempfile

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_research_agent import WebResearchAgent
from tools.web_search import WebSearchTool
from tools.web_scraper import WebScraper
from tools.content_analyzer import ContentAnalyzer
from tools.news_aggregator import NewsAggregator
import main

class TestIntegration(unittest.TestCase):
    """
    Integration tests for the web research agent.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create mock AI model
        self.mock_ai_model = MagicMock()
        self.mock_ai_model.generate_text.return_value = "This is a mock AI response."
        
        # Create agent with mock implementations
        self.agent = WebResearchAgent(
            ai_model=self.mock_ai_model,
            use_mock=True
        )
    
    def test_end_to_end_research(self):
        """
        Test end-to-end research process.
        """
        query = "What is artificial intelligence?"
        
        results = self.agent.research(query, depth=1, max_sources=2)
        
        self.assertIn("query", results)
        self.assertIn("answer", results)
        self.assertIn("sources", results)
        self.assertEqual(results["query"], query)
        self.assertNotEqual(results["answer"], "")
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('web_research_agent.WebResearchAgent.research')
    def test_main_script(self, mock_research, mock_parse_args):
        """
        Test the main script.
        """
        # Mock command line arguments
        mock_args = MagicMock()
        mock_args.query = "What is quantum computing?"
        mock_args.model = "gpt-4o"
        mock_args.depth = 2
        mock_args.max_sources = 5
        mock_args.output = "test_results.json"
        mock_args.search_api_key = None
        mock_args.news_api_key = None
        mock_args.ai_api_key = None
        mock_args.use_mock = True
        
        mock_parse_args.return_value = mock_args
        
        # Mock research results
        mock_research.return_value = {
            "query": "What is quantum computing?",
            "answer": "Quantum computing is a type of computing that uses quantum bits.",
            "sources": [
                {"title": "Source 1", "url": "https://example.com/1", "relevance": 0.9},
                {"title": "Source 2", "url": "https://example.com/2", "relevance": 0.8}
            ],
            "research_time": 1.5,
            "search_queries": ["What is quantum computing?"]
        }
        
        # Create a temporary file for output
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            mock_args.output = temp_file.name
            
            # Run the main function
            main.main()
            
            # Verify research was called
            mock_research.assert_called_once_with(
                query="What is quantum computing?",
                depth=2,
                max_sources=5
            )
            
            # Verify output file was created
            with open(temp_file.name, 'r') as f:
                results = json.load(f)
                self.assertEqual(results["query"], "What is quantum computing?")
                self.assertEqual(results["answer"], "Quantum computing is a type of computing that uses quantum bits.")
                self.assertEqual(len(results["sources"]), 2)
        
        # Clean up
        os.unlink(temp_file.name)
    
    def test_integration_with_real_tools(self):
        """
        Test integration with real tools (but still using mock APIs).
        """
        # Create real instances of tools but with mock APIs
        search_tool = WebSearchTool(use_mock=True)
        scraper = WebScraper()
        content_analyzer = ContentAnalyzer()
        news_aggregator = NewsAggregator(use_mock=True)
        
        # Create agent with real tools but mock APIs
        agent = WebResearchAgent(
            ai_model=self.mock_ai_model,
            search_api_key=None,
            news_api_key=None,
            use_mock=True
        )
        
        # Override the agent's tools with our instances
        agent.search_tool = search_tool
        agent.scraper = scraper
        agent.content_analyzer = content_analyzer
        agent.news_aggregator = news_aggregator
        
        # Test research with real tools
        query = "What is quantum computing?"
        results = agent.research(query, depth=1, max_sources=2)
        
        # Verify results
        self.assertEqual(results["query"], query)
        self.assertIn("answer", results)
        self.assertIn("sources", results)
        self.assertTrue(len(results["sources"]) > 0)


if __name__ == "__main__":
    unittest.main()
