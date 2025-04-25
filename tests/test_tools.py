import unittest
from unittest.mock import MagicMock, patch, ANY
import sys
import os
import json
import datetime
import requests
from bs4 import BeautifulSoup

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.web_search import WebSearchTool
from tools.web_scraper import WebScraper
from tools.content_analyzer import ContentAnalyzer
from tools.news_aggregator import NewsAggregator

class TestWebSearchTool(unittest.TestCase):
    """
    Test cases for the WebSearchTool.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        self.search_tool = WebSearchTool(use_mock=True)
        self.real_search_tool = WebSearchTool(api_key="test_key", use_mock=False)
    
    def test_search(self):
        """
        Test search functionality.
        """
        query = "test query"
        results = self.search_tool.search(query)
        
        self.assertIn("organic_results", results)
        self.assertTrue(len(results["organic_results"]) > 0)
        
        # Check result structure
        result = results["organic_results"][0]
        self.assertIn("title", result)
        self.assertIn("link", result)
        self.assertIn("snippet", result)
    
    def test_get_next_page(self):
        """
        Test pagination functionality.
        """
        query = "test query"
        first_page = self.search_tool.search(query)
        second_page = self.search_tool.get_next_page(first_page)
        
        self.assertIn("organic_results", second_page)
    
    @patch('requests.get')
    def test_real_search(self, mock_get):
        """
        Test real search with API.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "search_metadata": {"status": "Success"},
            "organic_results": [
                {"title": "Result 1", "link": "https://example.com/1", "snippet": "Snippet 1"},
                {"title": "Result 2", "link": "https://example.com/2", "snippet": "Snippet 2"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Test search
        results = self.real_search_tool.search("python programming")
        
        # Verify request
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], "https://serpapi.com/search")
        self.assertEqual(kwargs["params"]["q"], "python programming")
        self.assertEqual(kwargs["params"]["api_key"], "test_key")
        
        # Verify results
        self.assertEqual(len(results["organic_results"]), 2)
    
    @patch('requests.get')
    def test_search_error_handling(self, mock_get):
        """
        Test search error handling.
        """
        # Mock error response
        mock_get.side_effect = requests.RequestException("API error")
        
        # Test search with error
        results = self.real_search_tool.search("python programming")
        
        # Should fall back to mock implementation
        self.assertIn("organic_results", results)
        self.assertTrue(len(results["organic_results"]) > 0)
        self.assertTrue(all("mock" in result["title"].lower() for result in results["organic_results"]))


class TestWebScraper(unittest.TestCase):
    """
    Test cases for the WebScraper.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        self.scraper = WebScraper()
        self.scraper_no_robots = WebScraper(respect_robots=False)
    
    @patch('requests.get')
    def test_scrape(self, mock_get):
        """
        Test scraping functionality.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.text = """
        <html>
            <head>
                <title>Test Page</title>
                <meta name="description" content="Test description">
            </head>
            <body>
                <main>
                    <h1>Test Heading</h1>
                    <p>Test paragraph.</p>
                    <ul>
                        <li>Item 1</li>
                        <li>Item 2</li>
                    </ul>
                    <table>
                        <tr><th>Header 1</th><th>Header 2</th></tr>
                        <tr><td>Data 1</td><td>Data 2</td></tr>
                    </table>
                </main>
            </body>
        </html>
        """
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Test scraping
        result = self.scraper.scrape("https://example.com")
        
        # Verify request
        mock_get.assert_called_with("https://example.com", headers=ANY, timeout=10)
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["status_code"], 200)
        
        # Check content structure
        content = result["content"]
        self.assertEqual(content["title"], "Test Page")
        self.assertIn("Test paragraph", content["main_content"])
        self.assertEqual(len(content["headings"]), 1)
        self.assertEqual(content["headings"][0]["text"], "Test Heading")
        self.assertEqual(len(content["lists"]), 1)
        self.assertEqual(len(content["lists"][0]["items"]), 2)
        self.assertEqual(len(content["tables"]), 1)
        self.assertEqual(content["tables"][0]["headers"], ["Header 1", "Header 2"])
        self.assertEqual(len(content["tables"][0]["data"]), 1)
        self.assertEqual(content["metadata"]["description"], "Test description")
    
    @patch('requests.get')
    def test_scrape_error_handling(self, mock_get):
        """
        Test scraping error handling.
        """
        # Mock error response
        mock_get.side_effect = requests.RequestException("Connection error")
        
        # Test scraping with error
        result = self.scraper.scrape("https://example.com")
        
        # Verify result
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Connection error")
        self.assertEqual(result["content"], {})
    
    @patch('requests.get')
    def test_robots_txt_respect(self, mock_get):
        """
        Test robots.txt respect.
        """
        # Mock robots.txt response
        robots_response = MagicMock()
        robots_response.text = """
        User-agent: *
        Disallow: /private/
        """
        robots_response.status_code = 200
        
        # Mock page response
        page_response = MagicMock()
        page_response.text = "<html><body>Test</body></html>"
        page_response.status_code = 200
        
        # Set up mock to return different responses
        def side_effect(url, **kwargs):
            if "robots.txt" in url:
                return robots_response
            return page_response
        
        mock_get.side_effect = side_effect
        
        # Test allowed URL
        allowed_result = self.scraper.scrape("https://example.com/public")
        self.assertTrue(allowed_result["success"])
        
        # Test disallowed URL
        disallowed_result = self.scraper.scrape("https://example.com/private/page")
        self.assertFalse(disallowed_result["success"])
        self.assertIn("robots.txt", disallowed_result["error"])
        
        # Test with robots.txt respect disabled
        no_robots_result = self.scraper_no_robots.scrape("https://example.com/private/page")
        self.assertTrue(no_robots_result["success"])
    
    def test_check_robots_rules(self):
        """
        Test robots.txt rule checking.
        """
        robots_txt = """
        User-agent: *
        Disallow: /private/
        Disallow: /admin/
        
        User-agent: Googlebot
        Disallow: /google-only/
        """
        
        # Test allowed path
        self.assertTrue(self.scraper._check_robots_rules(robots_txt, "/public"))
        
        # Test disallowed path
        self.assertFalse(self.scraper._check_robots_rules(robots_txt, "/private/page"))
        self.assertFalse(self.scraper._check_robots_rules(robots_txt, "/admin/dashboard"))
        
        # Test specific user agent rule
        # Our scraper doesn't identify as Googlebot, so this should be allowed
        self.assertTrue(self.scraper._check_robots_rules(robots_txt, "/google-only/page"))


class TestContentAnalyzer(unittest.TestCase):
    """
    Test cases for the ContentAnalyzer.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        self.analyzer = ContentAnalyzer()
        
        # Create mock AI model
        self.mock_ai_model = MagicMock()
        self.mock_ai_model.generate_text.return_value = json.dumps({
            "relevance_assessment": 8,
            "key_facts": ["Fact 1", "Fact 2"],
            "potential_biases": "No significant biases detected",
            "confidence": 7
        })
        
        self.analyzer_with_ai = ContentAnalyzer(ai_model=self.mock_ai_model)
        
        # Test content
        self.test_content = {
            "title": "Python Programming",
            "main_content": "Python is a programming language. It was created by Guido van Rossum in 1991. Python is known for its simplicity and readability.",
            "headings": [
                {"level": 1, "text": "Python Programming"},
                {"level": 2, "text": "History of Python"}
            ],
            "tables": [],
            "lists": [],
            "metadata": {"datePublished": "2023-01-01T12:00:00Z"},
            "full_text": "Python is a programming language. It was created by Guido van Rossum in 1991. Python is known for its simplicity and readability."
        }
    
    def test_analyze(self):
        """
        Test content analysis.
        """
        query = "What is Python?"
        
        analysis = self.analyzer.analyze(self.test_content, query)
        
        self.assertIn("relevance_score", analysis)
        self.assertIn("information_density", analysis)
        self.assertIn("recency", analysis)
        self.assertIn("reliability", analysis)
        self.assertIn("key_sentences", analysis)
        
        # Check relevance score
        self.assertTrue(0 <= analysis["relevance_score"] <= 1)
        self.assertGreater(analysis["relevance_score"], 0.5)  # Should be relevant to the query
        
        # Check key sentences
        self.assertTrue(len(analysis["key_sentences"]) > 0)
        self.assertIn("Python is a programming language", analysis["key_sentences"][0])
        
        # Check recency
        self.assertTrue(analysis["recency"]["found"])
        self.assertEqual(analysis["recency"]["date"], "2023-01-01T12:00:00Z")
    
    def test_analyze_with_ai(self):
        """
        Test content analysis with AI model.
        """
        query = "What is Python?"
        
        analysis = self.analyzer_with_ai.analyze(self.test_content, query)
        
        self.assertIn("ai_analysis", analysis)
        self.assertTrue(len(analysis["ai_analysis"]) > 0)
        
        # Verify AI model was called
        self.mock_ai_model.generate_text.assert_called_once()
    
    def test_calculate_relevance(self):
        """
        Test relevance calculation.
        """
        # Highly relevant query
        relevant_query = "what is python"
        relevant_score = self.analyzer._calculate_relevance(self.test_content, relevant_query)
        
        # Less relevant query
        less_relevant_query = "java tutorial"
        less_relevant_score = self.analyzer._calculate_relevance(self.test_content, less_relevant_query)
        
        # Verify scores
        self.assertGreater(relevant_score, 0.5)
        self.assertLess(less_relevant_score, 0.5)
        self.assertGreater(relevant_score, less_relevant_score)
        
        # Test with query terms in title
        title_query = "python programming"
        title_score = self.analyzer._calculate_relevance(self.test_content, title_query)
        
        self.assertGreater(title_score, relevant_score)
    
    def test_calculate_information_density(self):
        """
        Test information density calculation.
        """
        # Test with normal content
        density = self.analyzer._calculate_information_density(self.test_content)
        
        self.assertTrue(0 <= density <= 1)
        
        # Test with empty content
        empty_content = {"main_content": "", "full_text": ""}
        empty_density = self.analyzer._calculate_information_density(empty_content)
        
        self.assertEqual(empty_density, 0.0)
        
        # Test with content containing tables and lists
        rich_content = {
            "main_content": self.test_content["main_content"],
            "full_text": self.test_content["full_text"],
            "tables": [{"headers": ["Col1", "Col2"], "data": [["A", "B"], ["C", "D"]]}],
            "lists": [{"type": "ul", "items": ["Item 1", "Item 2", "Item 3"]}],
            "headings": self.test_content["headings"]
        }
        
        rich_density = self.analyzer._calculate_information_density(rich_content)
        
        self.assertGreater(rich_density, density)
    
    def test_estimate_recency(self):
        """
        Test recency estimation.
        """
        # Test with metadata date
        recency = self.analyzer._estimate_recency(self.test_content)
        
        self.assertTrue(recency["found"])
        self.assertEqual(recency["date"], "2023-01-01T12:00:00Z")
        self.assertEqual(recency["source"], "datePublished")
        
        # Test with date in text
        text_date_content = {
            "main_content": "This article was published on January 15, 2023.",
            "full_text": "This article was published on January 15, 2023.",
            "metadata": {}
        }
        
        text_recency = self.analyzer._estimate_recency(text_date_content)
        
        self.assertTrue(text_recency["found"])
        self.assertEqual(text_recency["date_text"], "January 15, 2023")
        self.assertEqual(text_recency["source"], "text_pattern")
        
        # Test with no date
        no_date_content = {
            "main_content": "This is content with no date.",
            "full_text": "This is content with no date.",
            "metadata": {}
        }
        
        no_date_recency = self.analyzer._estimate_recency(no_date_content)
        
        self.assertFalse(no_date_recency["found"])
        self.assertIsNone(no_date_recency["source"])
    
    def test_estimate_reliability(self):
        """
        Test reliability estimation.
        """
        # Test with normal content
        reliability = self.analyzer._estimate_reliability(self.test_content)
        
        self.assertTrue(0 <= reliability <= 1)
        
        # Test with content containing citations
        cited_content = {
            "main_content": "According to Smith et al. [1], Python is popular. As referenced in Jones (2020), Python is used in data science.",
            "full_text": "According to Smith et al. [1], Python is popular. As referenced in Jones (2020), Python is used in data science."
        }
        
        cited_reliability = self.analyzer._estimate_reliability(cited_content)
        
        self.assertGreater(cited_reliability, reliability)
        
        # Test with content containing qualifying language
        qualifying_content = {
            "main_content": "Python may be the most popular language. It appears to be widely used in data science.",
            "full_text": "Python may be the most popular language. It appears to be widely used in data science."
        }
        
        qualifying_reliability = self.analyzer._estimate_reliability(qualifying_content)
        
        self.assertGreater(qualifying_reliability, reliability)
    
    def test_extract_key_sentences(self):
        """
        Test key sentence extraction.
        """
        query = "What is Python?"
        
        sentences = self.analyzer._extract_key_sentences(self.test_content, query)
        
        self.assertTrue(len(sentences) > 0)
        self.assertIn("Python is a programming language", sentences[0])
        
        # Test with different query
        history_query = "When was Python created?"
        
        history_sentences = self.analyzer._extract_key_sentences(self.test_content, history_query)
        
        self.assertTrue(len(history_sentences) > 0)
        self.assertIn("created by Guido van Rossum in 1991", history_sentences[0])


class TestNewsAggregator(unittest.TestCase):
    """
    Test cases for the NewsAggregator.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        self.news_aggregator = NewsAggregator(use_mock=True)
        self.real_news_aggregator = NewsAggregator(api_key="test_key", use_mock=False)
    
    def test_search_news(self):
        """
        Test news search functionality.
        """
        query = "artificial intelligence"
        results = self.news_aggregator.search_news(query)
        
        self.assertIn("articles", results)
        self.assertTrue(len(results["articles"]) > 0)
        
        # Check article structure
        article = results["articles"][0]
        self.assertIn("title", article)
        self.assertIn("url", article)
        self.assertIn("publishedAt", article)
        self.assertIn("description", article)
    
    @patch('requests.get')
    def test_real_search_news(self, mock_get):
        """
        Test real news search with API.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "title": "AI News 1",
                    "url": "https://example.com/news/1",
                    "description": "Description 1",
                    "publishedAt": "2023-01-01T12:00:00Z"
                },
                {
                    "title": "AI News 2",
                    "url": "https://example.com/news/2",
                    "description": "Description 2",
                    "publishedAt": "2023-01-02T12:00:00Z"
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Test search
        results = self.real_news_aggregator.search_news("artificial intelligence", days=7)
        
        # Verify request
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], "https://newsapi.org/v2/everything")
        self.assertEqual(kwargs["params"]["q"], "artificial intelligence")
        self.assertEqual(kwargs["params"]["language"], "en")
        self.assertEqual(kwargs["params"]["apiKey"], "test_key")
        
        # Verify results
        self.assertEqual(len(results["articles"]), 2)
        self.assertEqual(results["articles"][0]["title"], "AI News 1")
    
    @patch('requests.get')
    def test_news_search_error_handling(self, mock_get):
        """
        Test news search error handling.
        """
        # Mock error response
        mock_get.side_effect = requests.RequestException("API error")
        
        # Test search with error
        results = self.real_news_aggregator.search_news("artificial intelligence")
        
        # Should fall back to mock implementation
        self.assertIn("articles", results)
        self.assertTrue(len(results["articles"]) > 0)
        self.assertTrue(all("mock" in article["title"].lower() for article in results["articles"]))
    
    def test_filter_news(self):
        """
        Test news filtering functionality.
        """
        # Create test news results
        news_results = {
            "status": "ok",
            "totalResults": 3,
            "articles": [
                {
                    "title": "AI advances in healthcare",
                    "description": "New AI applications in healthcare",
                    "content": "Artificial intelligence is transforming healthcare."
                },
                {
                    "title": "Machine learning in finance",
                    "description": "ML applications in finance",
                    "content": "Machine learning algorithms are used in finance."
                },
                {
                    "title": "Sports news update",
                    "description": "Latest sports results",
                    "content": "Sports teams competed in tournaments."
                }
            ]
        }
        
        # Filter with keywords
        filtered_results = self.news_aggregator.filter_news(
            news_results,
            keywords=["ai", "intelligence"]
        )
        
        self.assertEqual(len(filtered_results["articles"]), 1)
        self.assertIn("AI", filtered_results["articles"][0]["title"])
        
        # Filter with exclude keywords
        excluded_results = self.news_aggregator.filter_news(
            news_results,
            exclude_keywords=["sports"]
        )
        
        self.assertEqual(len(excluded_results["articles"]), 2)
        self.assertNotIn("sports", excluded_results["articles"][0]["title"].lower())
        
        # Filter with minimum length
        length_results = self.news_aggregator.filter_news(
            news_results,
            min_length=100
        )
        
        self.assertEqual(len(length_results["articles"]), 0)  # All articles are shorter than 100 chars
    
    def test_get_trending_topics(self):
        """
        Test trending topics functionality.
        """
        topics = self.news_aggregator.get_trending_topics()
        
        self.assertTrue(len(topics) > 0)
        self.assertTrue(all(isinstance(topic, str) for topic in topics))
        
        # Test with specific category
        tech_topics = self.news_aggregator.get_trending_topics(category="technology")
        
        self.assertTrue(len(tech_topics) > 0)
        self.assertIn("Artificial Intelligence", tech_topics)
        
        # Test with different category
        health_topics = self.news_aggregator.get_trending_topics(category="health")
        
        self.assertTrue(len(health_topics) > 0)
        self.assertNotEqual(tech_topics, health_topics)


class TestAIModelWrapper(unittest.TestCase):
    """
    Test cases for the AIModelWrapper.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # We'll test with the mock implementation since we can't assume API access
        self.ai_wrapper = AIModelWrapper("test-model")
    
    def test_mock_generate_text(self):
        """
        Test mock text generation.
        """
        prompt = "What is Python?"
        
        response = self.ai_wrapper._mock_generate_text(prompt)
        
        self.assertTrue(isinstance(response, str))
        self.assertIn("mock response", response)
        self.assertIn(prompt[:50], response)
    
    @patch('openai.OpenAI')
    def test_openai_integration(self, mock_openai):
        """
        Test OpenAI integration.
        """
        # Skip if OpenAI is not available
        if not 'OPENAI_AVAILABLE' in globals() or not OPENAI_AVAILABLE:
            self.skipTest("OpenAI not available")
        
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "OpenAI response"
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Create wrapper with OpenAI model
        openai_wrapper = AIModelWrapper("gpt-4", api_key="test_key")
        
        # Test generate_text
        response = openai_wrapper.generate_text("What is Python?")
        
        # Verify OpenAI was called
        mock_client.chat.completions.create.assert_called_once()
        self.assertEqual(response, "OpenAI response")
    
    @patch('google.generativeai.GenerativeModel')
    def test_gemini_integration(self, mock_gemini_model):
        """
        Test Gemini integration.
        """
        # Skip if Gemini is not available
        if not 'GEMINI_AVAILABLE' in globals() or not GEMINI_AVAILABLE:
            self.skipTest("Gemini not available")
        
        # Mock Gemini model
        mock_model = MagicMock()
        mock_gemini_model.return_value = mock_model
        
        mock_response = MagicMock()
        mock_response.text = "Gemini response"
        
        mock_model.generate_content.return_value = mock_response
        
        # Create wrapper with Gemini model
        gemini_wrapper = AIModelWrapper("gemini-pro", api_key="test_key")
        
        # Test generate_text
        response = gemini_wrapper.generate_text("What is Python?")
        
        # Verify Gemini was called
        mock_model.generate_content.assert_called_once()
        self.assertEqual(response, "Gemini response")
    
    @patch('anthropic.Anthropic')
    def test_anthropic_integration(self, mock_anthropic):
        """
        Test Anthropic integration.
        """
        # Skip if Anthropic is not available
        if not 'ANTHROPIC_AVAILABLE' in globals() or not ANTHROPIC_AVAILABLE:
            self.skipTest("Anthropic not available")
        
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        mock_message = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Claude response"
        mock_message.content = [mock_content]
        
        mock_client.messages.create.return_value = mock_message
        
        # Create wrapper with Claude model
        claude_wrapper = AIModelWrapper("claude-3-opus", api_key="test_key")
        
        # Test generate_text
        response = claude_wrapper.generate_text("What is Python?")
        
        # Verify Anthropic was called
        mock_client.messages.create.assert_called_once()
        self.assertEqual(response, "Claude response")


if __name__ == "__main__":
    unittest.main()
