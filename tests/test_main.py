import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
import tempfile

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from main import AIModelWrapper

class TestMain(unittest.TestCase):
    """
    Test cases for the main script and AIModelWrapper.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        pass
    
    def test_ai_model_wrapper_init(self):
        """
        Test AIModelWrapper initialization.
        """
        # Test with mock model
        wrapper = AIModelWrapper("test-model")
        self.assertEqual(wrapper.model_name, "test-model")
        self.assertEqual(wrapper.provider, "mock")
        
        # Test with OpenAI model name
        with patch('main.OPENAI_AVAILABLE', True):
            with patch('main.OpenAI') as mock_openai:
                wrapper = AIModelWrapper("gpt-4")
                self.assertEqual(wrapper.provider, "openai")
                mock_openai.assert_called_once()
        
        # Test with Gemini model name
        with patch('main.GEMINI_AVAILABLE', True):
            with patch('main.genai') as mock_genai:
                wrapper = AIModelWrapper("gemini-pro")
                self.assertEqual(wrapper.provider, "gemini")
                mock_genai.configure.assert_called_once()
        
        # Test with Claude model name
        with patch('main.ANTHROPIC_AVAILABLE', True):
            with patch('main.anthropic.Anthropic') as mock_anthropic:
                wrapper = AIModelWrapper("claude-3")
                self.assertEqual(wrapper.provider, "anthropic")
                mock_anthropic.assert_called_once()
    
    def test_generate_text(self):
        """
        Test text generation.
        """
        # Test mock implementation
        wrapper = AIModelWrapper("test-model")
        text = wrapper.generate_text("Test prompt")
        self.assertIn("mock response", text)
        self.assertIn("Test prompt", text)
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('web_research_agent.WebResearchAgent')
    def test_main_function(self, mock_agent_class, mock_parse_args):
        """
        Test the main function.
        """
        # Mock command line arguments
        mock_args = MagicMock()
        mock_args.query = "What is quantum computing?"
        mock_args.model = "test-model"
        mock_args.depth = 2
        mock_args.max_sources = 5
        mock_args.output = "test_output.json"
        mock_args.search_api_key = "test_search_key"
        mock_args.news_api_key = "test_news_key"
        mock_args.ai_api_key = "test_ai_key"
        mock_args.use_mock = True
        
        mock_parse_args.return_value = mock_args
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        # Mock research results
        mock_agent.research.return_value = {
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
            
            # Verify agent was created with correct parameters
            mock_agent_class.assert_called_once()
            _, kwargs = mock_agent_class.call_args
            self.assertEqual(kwargs["search_api_key"], "test_search_key")
            self.assertEqual(kwargs["news_api_key"], "test_news_key")
            self.assertEqual(kwargs["use_mock"], True)
            
            # Verify research was called with correct parameters
            mock_agent.research.assert_called_once_with(
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


if __name__ == "__main__":
    unittest.main()
