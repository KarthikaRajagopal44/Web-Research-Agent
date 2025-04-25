import unittest
from unittest.mock import MagicMock, patch, ANY
import sys
import os
import json
import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_research_agent import WebResearchAgent
from tools.web_search import WebSearchTool
from tools.web_scraper import WebScraper
from tools.content_analyzer import ContentAnalyzer
from tools.news_aggregator import NewsAggregator

class TestWebResearchAgent(unittest.TestCase):
    """
    Test cases for the WebResearchAgent.
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
        
        # Create test data
        self.test_query = "What is quantum computing?"
        self.test_news_query = "Latest developments in AI technology"
        self.test_howto_query = "How to build a machine learning model"
        self.test_exploratory_query = "Impact of climate change on agriculture"
    
    def test_analyze_query(self):
        """
        Test query analysis functionality.
        """
        # Test factual query
        factual_analysis = self.agent._analyze_query(self.test_query)
        
        self.assertEqual(factual_analysis["query_type"], "factual")
        self.assertIn("quantum computing", factual_analysis["key_concepts"])
        
        # Test news query
        news_analysis = self.agent._analyze_query(self.test_news_query)
        
        self.assertEqual(news_analysis["query_type"], "news")
        self.assertTrue(news_analysis["time_sensitive"])
        self.assertIn("ai technology", news_analysis["key_concepts"])
        
        # Test how-to query
        howto_analysis = self.agent._analyze_query(self.test_howto_query)
        
        self.assertEqual(howto_analysis["query_type"], "how-to")
        self.assertIn("machine learning model", howto_analysis["key_concepts"])
        
        # Test exploratory query
        exploratory_analysis = self.agent._analyze_query(self.test_exploratory_query)
        
        self.assertEqual(exploratory_analysis["query_type"], "exploratory")
        self.assertIn("climate change", exploratory_analysis["key_concepts"])
        self.assertIn("agriculture", exploratory_analysis["key_concepts"])
    
    def test_determine_query_type(self):
        """
        Test query type determination.
        """
        # Test factual queries
        self.assertEqual(self.agent._determine_query_type("What is Python?"), "factual")
        self.assertEqual(self.agent._determine_query_type("Who is Albert Einstein?"), "factual")
        self.assertEqual(self.agent._determine_query_type("When did World War II end?"), "factual")
        self.assertEqual(self.agent._determine_query_type("Define artificial intelligence"), "factual")
        
        # Test news queries
        self.assertEqual(self.agent._determine_query_type("Latest news on COVID-19"), "news")
        self.assertEqual(self.agent._determine_query_type("Recent developments in quantum computing"), "news")
        self.assertEqual(self.agent._determine_query_type("Current state of global economy"), "news")
        
        # Test how-to queries
        self.assertEqual(self.agent._determine_query_type("How to build a website"), "how-to")
        self.assertEqual(self.agent._determine_query_type("Steps to learn programming"), "how-to")
        self.assertEqual(self.agent._determine_query_type("Guide for machine learning"), "how-to")
        
        # Test opinion queries
        self.assertEqual(self.agent._determine_query_type("Best programming languages to learn"), "opinion")
        self.assertEqual(self.agent._determine_query_type("Pros and cons of remote work"), "opinion")
        self.assertEqual(self.agent._determine_query_type("Should I learn Python or JavaScript?"), "opinion")
        
        # Test exploratory queries (default)
        self.assertEqual(self.agent._determine_query_type("Quantum computing"), "exploratory")
        self.assertEqual(self.agent._determine_query_type("Climate change effects"), "exploratory")
    
    def test_identify_information_type(self):
        """
        Test information type identification.
        """
        # Test news information
        self.assertEqual(self.agent._identify_information_type("Latest developments in AI"), "news")
        self.assertEqual(self.agent._identify_information_type("Recent updates on climate change"), "news")
        
        # Test historical information
        self.assertEqual(self.agent._identify_information_type("History of computing"), "historical")
        self.assertEqual(self.agent._identify_information_type("Evolution of the internet"), "historical")
        
        # Test technical information
        self.assertEqual(self.agent._identify_information_type("Technical aspects of quantum computing"), "technical")
        self.assertEqual(self.agent._identify_information_type("Programming in Python"), "technical")
        
        # Test opinion information
        self.assertEqual(self.agent._identify_information_type("Best programming languages"), "opinions")
        self.assertEqual(self.agent._identify_information_type("Reviews of AI tools"), "opinions")
        
        # Test default (facts)
        self.assertEqual(self.agent._identify_information_type("Quantum computing"), "facts")
        self.assertEqual(self.agent._identify_information_type("Climate change"), "facts")
    
    def test_extract_key_concepts(self):
        """
        Test key concept extraction.
        """
        # Test simple query
        concepts = self.agent._extract_key_concepts("Quantum computing")
        self.assertIn("quantum", concepts)
        self.assertIn("computing", concepts)
        self.assertIn("quantum computing", concepts)
        
        # Test complex query
        concepts = self.agent._extract_key_concepts("What is the impact of artificial intelligence on healthcare?")
        self.assertIn("artificial intelligence", concepts)
        self.assertIn("healthcare", concepts)
        self.assertIn("impact", concepts)
        
        # Test query with stop words
        concepts = self.agent._extract_key_concepts("How does the internet work?")
        self.assertIn("internet", concepts)
        self.assertIn("work", concepts)
        self.assertNotIn("how", concepts)
        self.assertNotIn("does", concepts)
        self.assertNotIn("the", concepts)
    
    def test_is_time_sensitive(self):
        """
        Test time sensitivity detection.
        """
        # Test time-sensitive queries
        self.assertTrue(self.agent._is_time_sensitive("Latest developments in AI"))
        self.assertTrue(self.agent._is_time_sensitive("Recent news on climate change"))
        self.assertTrue(self.agent._is_time_sensitive("Current state of quantum computing"))
        self.assertTrue(self.agent._is_time_sensitive("Updates on COVID-19"))
        
        # Test non-time-sensitive queries
        self.assertFalse(self.agent._is_time_sensitive("What is quantum computing?"))
        self.assertFalse(self.agent._is_time_sensitive("How does the internet work?"))
        self.assertFalse(self.agent._is_time_sensitive("History of artificial intelligence"))
    
    def test_generate_search_queries(self):
        """
        Test search query generation.
        """
        # Test factual query
        query = "What is quantum computing?"
        query_analysis = {
            "query_type": "factual",
            "information_type": "facts",
            "key_concepts": ["quantum computing", "quantum", "computing"],
            "time_sensitive": False
        }
        
        search_queries = self.agent._generate_search_queries(query, query_analysis)
        
        self.assertIn(query, search_queries)  # Original query should be included
        self.assertTrue(any("what is quantum computing" in q.lower() for q in search_queries))
        self.assertTrue(any("quantum computing definition" in q.lower() for q in search_queries))
        
        # Test news query
        query = "Latest developments in AI"
        query_analysis = {
            "query_type": "news",
            "information_type": "news",
            "key_concepts": ["developments", "latest developments", "ai"],
            "time_sensitive": True
        }
        
        search_queries = self.agent._generate_search_queries(query, query_analysis)
        
        self.assertIn(query, search_queries)
        self.assertTrue(any("latest news" in q.lower() for q in search_queries))
        self.assertTrue(any("recent developments" in q.lower() for q in search_queries))
        
        # Test how-to query
        query = "How to build a machine learning model"
        query_analysis = {
            "query_type": "how-to",
            "information_type": "technical",
            "key_concepts": ["machine learning model", "build", "machine learning"],
            "time_sensitive": False
        }
        
        search_queries = self.agent._generate_search_queries(query, query_analysis)
        
        self.assertIn(query, search_queries)
        self.assertTrue(any("guide" in q.lower() for q in search_queries))
        self.assertTrue(any("tutorial" in q.lower() for q in search_queries))
        
        # Test with time-sensitive flag
        query_analysis["time_sensitive"] = True
        time_queries = self.agent._generate_search_queries(query, query_analysis)
        
        self.assertTrue(any("latest" in q.lower() for q in time_queries))
        self.assertTrue(any("recent" in q.lower() for q in time_queries))
    
    @patch('tools.web_search.WebSearchTool.search')
    def test_perform_search(self, mock_search):
        """
        Test search functionality.
        """
        # Mock search results
        mock_search.return_value = {
            "organic_results": [
                {
                    "title": "What is Quantum Computing? - IBM",
                    "link": "https://www.ibm.com/quantum-computing/what-is-quantum-computing/",
                    "snippet": "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.",
                    "position": 1
                },
                {
                    "title": "Quantum Computing - Wikipedia",
                    "link": "https://en.wikipedia.org/wiki/Quantum_computing",
                    "snippet": "Quantum computing is the exploitation of collective properties of quantum states, such as superposition and entanglement, to perform computation.",
                    "position": 2
                }
            ]
        }
        
        query = "What is quantum computing?"
        query_analysis = self.agent._analyze_query(query)
        
        results = self.agent._perform_search(query, query_analysis)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "What is Quantum Computing? - IBM")
        self.assertEqual(results[0]["link"], "https://www.ibm.com/quantum-computing/what-is-quantum-computing/")
        self.assertEqual(results[0]["source"], "web")
        
        # Test news search
        query_analysis["query_type"] = "news"
        query_analysis["information_type"] = "news"
        
        with patch('tools.news_aggregator.NewsAggregator.search_news') as mock_news_search:
            mock_news_search.return_value = {
                "articles": [
                    {
                        "title": "Latest Quantum Computing Breakthrough",
                        "url": "https://example.com/news/1",
                        "description": "Scientists achieve quantum supremacy.",
                        "publishedAt": "2023-01-01T12:00:00Z"
                    }
                ]
            }
            
            results = self.agent._perform_search(query, query_analysis)
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["title"], "Latest Quantum Computing Breakthrough")
            self.assertEqual(results[0]["link"], "https://example.com/news/1")
            self.assertEqual(results[0]["source"], "news")
    
    def test_deduplicate_results(self):
        """
        Test result deduplication.
        """
        # Create test results with duplicates
        results = [
            {"title": "Result 1", "link": "https://example.com/page1"},
            {"title": "Result 2", "link": "https://example.com/page2"},
            {"title": "Result 3", "link": "https://example.com/page1"},  # Duplicate
            {"title": "Result 4", "link": "https://example.org/page1"}
        ]
        
        unique_results = self.agent._deduplicate_results(results)
        
        self.assertEqual(len(unique_results), 3)  # Should remove 1 duplicate
        
        # Test with different URL formats but same content
        results = [
            {"title": "Result 1", "link": "https://example.com/page1"},
            {"title": "Result 2", "link": "https://example.com/page2"},
            {"title": "Result 3", "link": "https://example.com/page1?utm_source=google"},  # Same as first but with query params
            {"title": "Result 4", "link": "https://example.org/page1"}
        ]
        
        unique_results = self.agent._deduplicate_results(results)
        
        self.assertEqual(len(unique_results), 3)  # Should normalize URLs and remove duplicates
    
    def test_rank_results(self):
        """
        Test result ranking.
        """
        query_analysis = {
            "key_concepts": ["python", "programming"],
            "time_sensitive": False
        }
        
        results = [
            {"title": "Java Tutorial", "snippet": "Learn Java programming", "position": 1},
            {"title": "Python Programming", "snippet": "Learn Python programming", "position": 3},
            {"title": "Python for Beginners", "snippet": "Python tutorial for beginners", "position": 2}
        ]
        
        ranked_results = self.agent._rank_results(results, query_analysis)
        
        # Python-related results should be ranked higher
        self.assertEqual(ranked_results[0]["title"], "Python Programming")
        self.assertEqual(ranked_results[1]["title"], "Python for Beginners")
        self.assertEqual(ranked_results[2]["title"], "Java Tutorial")
        
        # Test with time-sensitive query
        query_analysis["time_sensitive"] = True
        
        results = [
            {"title": "Java Tutorial", "snippet": "Learn Java programming", "source": "web", "position": 1},
            {"title": "Python Programming", "snippet": "Learn Python programming", "source": "web", "position": 2},
            {"title": "Latest Python News", "snippet": "Recent developments in Python", "source": "news", "position": 3}
        ]
        
        ranked_results = self.agent._rank_results(results, query_analysis)
        
        # News source should be ranked higher for time-sensitive query
        self.assertEqual(ranked_results[0]["title"], "Latest Python News")
    
    @patch('tools.web_scraper.WebScraper.scrape')
    @patch('tools.content_analyzer.ContentAnalyzer.analyze')
    def test_extract_and_analyze_content(self, mock_analyze, mock_scrape):
        """
        Test content extraction and analysis.
        """
        # Mock scraper and analyzer responses
        mock_scrape.return_value = {
            "success": True,
            "status_code": 200,
            "content": {
                "title": "Test Page",
                "main_content": "This is test content about Python programming.",
                "headings": [{"level": 1, "text": "Python Programming"}],
                "tables": [],
                "lists": [],
                "metadata": {},
                "full_text": "This is test content about Python programming."
            }
        }
        
        mock_analyze.return_value = {
            "relevance_score": 0.8,
            "information_density": 0.6,
            "recency": {"found": False},
            "reliability": 0.7,
            "key_sentences": ["This is test content about Python programming."],
            "ai_analysis": {}
        }
        
        result = {"title": "Test Page", "link": "https://example.com/test"}
        query = "Python programming"
        
        analyzed_content = self.agent._extract_and_analyze_content(result, query)
        
        self.assertIsNotNone(analyzed_content)
        self.assertEqual(analyzed_content["url"], "https://example.com/test")
        self.assertEqual(analyzed_content["title"], "Test Page")
        self.assertEqual(analyzed_content["relevance_score"], 0.8)
        self.assertEqual(analyzed_content["reliability"], 0.7)
        self.assertEqual(analyzed_content["key_sentences"], ["This is test content about Python programming."])
        
        # Test with scraper failure
        mock_scrape.return_value = {
            "success": False,
            "error": "Connection error",
            "content": {}
        }
        
        analyzed_content = self.agent._extract_and_analyze_content(result, query)
        
        self.assertIsNone(analyzed_content)
    
    def test_group_similar_information(self):
        """
        Test grouping of similar information.
        """
        key_sentences = [
            {
                "sentence": "Python is a programming language.",
                "source_url": "https://example.com/1",
                "source_title": "Source 1",
                "relevance_score": 0.8,
                "reliability": 0.7
            },
            {
                "sentence": "Python is a high-level programming language.",
                "source_url": "https://example.com/2",
                "source_title": "Source 2",
                "relevance_score": 0.7,
                "reliability": 0.8
            },
            {
                "sentence": "Java is a popular programming language.",
                "source_url": "https://example.com/3",
                "source_title": "Source 3",
                "relevance_score": 0.6,
                "reliability": 0.7
            }
        ]
        
        grouped_info = self.agent._group_similar_information(key_sentences)
        
        self.assertEqual(len(grouped_info), 2)  # Should group the two Python sentences
        
        # First group should have 2 sentences (the Python ones)
        self.assertEqual(len(grouped_info[0]["sentences"]), 2)
        self.assertEqual(len(grouped_info[0]["sources"]), 2)
        
        # Second group should have 1 sentence (the Java one)
        self.assertEqual(len(grouped_info[1]["sentences"]), 1)
        self.assertEqual(len(grouped_info[1]["sources"]), 1)
    
    def test_identify_contradictions(self):
        """
        Test contradiction identification.
        """
        grouped_information = [
            {
                "sentences": [
                    {
                        "sentence": "The study shows an increase in global temperatures.",
                        "source_url": "https://example.com/1",
                        "source_title": "Source 1",
                        "relevance_score": 0.8,
                        "reliability": 0.7
                    }
                ],
                "sources": ["https://example.com/1"],
                "reliability": 0.7
            },
            {
                "sentences": [
                    {
                        "sentence": "Recent data indicates a decrease in global temperatures.",
                        "source_url": "https://example.com/2",
                        "source_title": "Source 2",
                        "relevance_score": 0.7,
                        "reliability": 0.6
                    }
                ],
                "sources": ["https://example.com/2"],
                "reliability": 0.6
            },
            {
                "sentences": [
                    {
                        "sentence": "Climate change is affecting ecosystems.",
                        "source_url": "https://example.com/3",
                        "source_title": "Source 3",
                        "relevance_score": 0.9,
                        "reliability": 0.8
                    }
                ],
                "sources": ["https://example.com/3"],
                "reliability": 0.8
            }
        ]
        
        contradictions = self.agent._identify_contradictions(grouped_information)
        
        self.assertEqual(len(contradictions), 1)  # Should identify one contradiction
        self.assertEqual(contradictions[0]["contradiction_type"], "increase vs decrease")
        self.assertEqual(contradictions[0]["group1"], grouped_information[0])
        self.assertEqual(contradictions[0]["group2"], grouped_information[1])
    
    def test_generate_summary(self):
        """
        Test summary generation.
        """
        grouped_information = [
            {
                "sentences": [
                    {
                        "sentence": "Python is a popular programming language.",
                        "source_url": "https://example.com/1",
                        "source_title": "Source 1",
                        "relevance_score": 0.8,
                        "reliability": 0.7
                    }
                ],
                "sources": ["https://example.com/1"],
                "reliability": 0.7
            },
            {
                "sentences": [
                    {
                        "sentence": "Python was created by Guido van Rossum.",
                        "source_url": "https://example.com/2",
                        "source_title": "Source 2",
                        "relevance_score": 0.7,
                        "reliability": 0.8
                    }
                ],
                "sources": ["https://example.com/2"],
                "reliability": 0.8
            }
        ]
        
        contradictions = []
        
        summary = self.agent._generate_summary(grouped_information, contradictions, "What is Python?")
        
        self.assertIn("Based on research from 2 sources", summary)
        self.assertIn("Python is a popular programming language", summary)
        self.assertIn("Python was created by Guido van Rossum", summary)
        
        # Test with contradictions
        contradictions = [
            {
                "group1": grouped_information[0],
                "group2": {
                    "sentences": [
                        {
                            "sentence": "Python is not widely used in industry.",
                            "source_url": "https://example.com/3",
                            "source_title": "Source 3",
                            "relevance_score": 0.6,
                            "reliability": 0.5
                        }
                    ],
                    "sources": ["https://example.com/3"],
                    "reliability": 0.5
                },
                "contradiction_type": "popular vs not"
            }
        ]
        
        summary = self.agent._generate_summary(grouped_information, contradictions, "What is Python?")
        
        self.assertIn("Contradictory information", summary)
        self.assertIn("Python is a popular programming language vs Python is not widely used in industry", summary)
    
    @patch('web_research_agent.WebResearchAgent._analyze_query')
    @patch('web_research_agent.WebResearchAgent._generate_search_queries')
    @patch('web_research_agent.WebResearchAgent._perform_search')
    @patch('web_research_agent.WebResearchAgent._deduplicate_results')
    @patch('web_research_agent.WebResearchAgent._rank_results')
    @patch('web_research_agent.WebResearchAgent._extract_and_analyze_content')
    @patch('web_research_agent.WebResearchAgent._synthesize_information')
    @patch('web_research_agent.WebResearchAgent._generate_answer')
    def test_research_end_to_end(self, mock_generate_answer, mock_synthesize, mock_extract, 
                                mock_rank, mock_deduplicate, mock_search, mock_gen_queries, mock_analyze):
        """
        Test the end-to-end research process.
        """
        # Set up mocks
        mock_analyze.return_value = {
            "query_type": "factual",
            "information_type": "facts",
            "key_concepts": ["quantum computing"],
            "time_sensitive": False
        }
        
        mock_gen_queries.return_value = ["What is quantum computing?", "quantum computing explained"]
        
        mock_search.return_value = [
            {"title": "Quantum Computing - IBM", "link": "https://example.com/1", "snippet": "About quantum computing"},
            {"title": "Quantum Computing - Wikipedia", "link": "https://example.com/2", "snippet": "Quantum computing is..."}
        ]
        
        mock_deduplicate.return_value = mock_search.return_value
        mock_rank.return_value = mock_search.return_value
        
        mock_extract.side_effect = [
            {
                "url": "https://example.com/1",
                "title": "Quantum Computing - IBM",
                "main_content": "Quantum computing uses quantum bits.",
                "relevance_score": 0.9,
                "reliability": 0.8,
                "key_sentences": ["Quantum computing uses quantum bits."]
            },
            {
                "url": "https://example.com/2",
                "title": "Quantum Computing - Wikipedia",
                "main_content": "Quantum computing is based on quantum mechanics.",
                "relevance_score": 0.8,
                "reliability": 0.9,
                "key_sentences": ["Quantum computing is based on quantum mechanics."]
            }
        ]
        
        mock_synthesize.return_value = {
            "key_information": [
                {"sentences": [{"sentence": "Quantum computing uses quantum bits."}], "sources": ["https://example.com/1"]},
                {"sentences": [{"sentence": "Quantum computing is based on quantum mechanics."}], "sources": ["https://example.com/2"]}
            ],
            "contradictions": [],
            "summary": "Quantum computing uses quantum bits and is based on quantum mechanics."
        }
        
        mock_generate_answer.return_value = "Quantum computing is a type of computing that uses quantum bits and is based on quantum mechanics."
        
        # Call the research method
        result = self.agent.research("What is quantum computing?", depth=1, max_sources=2)
        
        # Verify the result
        self.assertEqual(result["query"], "What is quantum computing?")
        self.assertEqual(result["answer"], "Quantum computing is a type of computing that uses quantum bits and is based on quantum mechanics.")
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["sources"][0]["url"], "https://example.com/1")
        self.assertEqual(result["sources"][1]["url"], "https://example.com/2")
        
        # Verify method calls
        mock_analyze.assert_called_once_with("What is quantum computing?")
        mock_gen_queries.assert_called_once()
        mock_search.assert_called()
        mock_deduplicate.assert_called_once()
        mock_rank.assert_called_once()
        mock_extract.assert_called()
        mock_synthesize.assert_called_once()
        mock_generate_answer.assert_called_once()
    
    def test_research_with_no_results(self):
        """
        Test research with no results.
        """
        # Create a new agent with mocked methods
        agent = WebResearchAgent(ai_model=self.mock_ai_model, use_mock=True)
        
        # Mock the _perform_search method to return empty results
        agent._perform_search = MagicMock(return_value=[])
        
        # Call the research method
        result = agent.research("Non-existent topic that should return no results")
        
        # Verify the result
        self.assertEqual(result["query"], "Non-existent topic that should return no results")
        self.assertIn("No information could be found", result["answer"])
        self.assertEqual(len(result["sources"]), 0)
    
    def test_research_with_ai_model(self):
        """
        Test research with AI model integration.
        """
        # Create a new agent with a real AI model mock
        ai_model = MagicMock()
        ai_model.generate_text.return_value = "AI-generated answer about quantum computing."
        
        agent = WebResearchAgent(ai_model=ai_model, use_mock=True)
        
        # Mock the necessary methods
        agent._analyze_query = MagicMock(return_value={
            "query_type": "factual",
            "information_type": "facts",
            "key_concepts": ["quantum computing"],
            "time_sensitive": False
        })
        
        agent._generate_search_queries = MagicMock(return_value=["What is quantum computing?"])
        agent._perform_search = MagicMock(return_value=[
            {"title": "Quantum Computing", "link": "https://example.com/1", "snippet": "About quantum computing"}
        ])
        agent._deduplicate_results = MagicMock(return_value=[
            {"title": "Quantum Computing", "link": "https://example.com/1", "snippet": "About quantum computing"}
        ])
        agent._rank_results = MagicMock(return_value=[
            {"title": "Quantum Computing", "link": "https://example.com/1", "snippet": "About quantum computing"}
        ])
        
        agent._extract_and_analyze_content = MagicMock(return_value={
            "url": "https://example.com/1",
            "title": "Quantum Computing",
            "main_content": "About quantum computing",
            "relevance_score": 0.9,
            "reliability": 0.8,
            "key_sentences": ["Quantum computing uses quantum bits."]
        })
        
        # Call the research method
        result = agent.research("What is quantum computing?")
        
        # Verify AI model was used
        ai_model.generate_text.assert_called()
        
        # Verify the result contains AI-generated content
        self.assertEqual(result["query"], "What is quantum computing?")
        self.assertIn("AI-generated", result["answer"])

if __name__ == "__main__":
    unittest.main()
