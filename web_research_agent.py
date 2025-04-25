import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import re
from urllib.parse import urlparse

# Import tools
from tools.web_search import WebSearchTool
from tools.web_scraper import WebScraper
from tools.content_analyzer import ContentAnalyzer
from tools.news_aggregator import NewsAggregator

class WebResearchAgent:
    """
    Web Research Agent that can analyze queries, search the web, extract content,
    and synthesize information to answer user questions.
    """
    
    def __init__(
        self,
        ai_model=None,
        search_api_key: Optional[str] = None,
        news_api_key: Optional[str] = None,
        use_mock: bool = False
    ):
        """
        Initialize the WebResearchAgent.
        
        Args:
            ai_model: AI model to use for analysis and synthesis
            search_api_key: API key for the search service
            news_api_key: API key for the news service
            use_mock: Whether to use mock implementations instead of real APIs
        """
        self.ai_model = ai_model
        self.use_mock = use_mock
        
        # Initialize tools
        self.search_tool = WebSearchTool(api_key=search_api_key, use_mock=use_mock)
        self.scraper = WebScraper(respect_robots=True)
        self.content_analyzer = ContentAnalyzer(ai_model=ai_model)
        self.news_aggregator = NewsAggregator(api_key=news_api_key, use_mock=use_mock)
        
        # Initialize state
        self.research_state = {}
        
    def research(self, query: str, depth: int = 2, max_sources: int = 5) -> Dict[str, Any]:
        """
        Perform web research to answer the query.
        
        Args:
            query: User's research query
            depth: Depth of research (1-3)
            max_sources: Maximum number of sources to analyze
            
        Returns:
            Dictionary containing research results and answer
        """
        # Initialize research state
        self.research_state = {
            "query": query,
            "depth": depth,
            "max_sources": max_sources,
            "start_time": time.time(),
            "search_queries": [],
            "search_results": [],
            "analyzed_sources": [],
            "key_information": [],
            "contradictions": [],
            "answer": "",
            "sources": []
        }
        
        # Step 1: Analyze the query
        query_analysis = self._analyze_query(query)
        self.research_state["query_analysis"] = query_analysis
        
        # Step 2: Generate search queries
        search_queries = self._generate_search_queries(query, query_analysis)
        self.research_state["search_queries"] = search_queries
        
        # Step 3: Perform searches
        all_search_results = []
        for search_query in search_queries:
            search_results = self._perform_search(search_query, query_analysis)
            all_search_results.extend(search_results)
            
            # Check if we have enough results
            if len(all_search_results) >= max_sources * 2:
                break
        
        # Deduplicate and rank search results
        unique_results = self._deduplicate_results(all_search_results)
        ranked_results = self._rank_results(unique_results, query_analysis)
        
        # Limit to max_sources
        top_results = ranked_results[:max_sources]
        self.research_state["search_results"] = top_results
        
        # Step 4: Extract and analyze content
        analyzed_sources = []
        for result in top_results:
            source_analysis = self._extract_and_analyze_content(result, query)
            if source_analysis:
                analyzed_sources.append(source_analysis)
        
        self.research_state["analyzed_sources"] = analyzed_sources
        
        # Step 5: Synthesize information
        synthesis = self._synthesize_information(analyzed_sources, query, query_analysis)
        
        # Step 6: Generate final answer
        answer = self._generate_answer(synthesis, query)
        
        # Update research state
        self.research_state["answer"] = answer
        self.research_state["sources"] = [source["url"] for source in analyzed_sources]
        self.research_state["end_time"] = time.time()
        self.research_state["duration"] = self.research_state["end_time"] - self.research_state["start_time"]
        
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "url": source["url"],
                    "title": source["title"],
                    "relevance": source["relevance_score"]
                }
                for source in analyzed_sources
            ],
            "research_time": self.research_state["duration"],
            "search_queries": search_queries
        }
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user query to understand intent and information needs.
        
        Args:
            query: User's research query
            
        Returns:
            Dictionary containing query analysis
        """
        # Determine query type
        query_type = self._determine_query_type(query)
        
        # Identify information type needed
        info_type = self._identify_information_type(query)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(query)
        
        # Check if query is time-sensitive
        time_sensitive = self._is_time_sensitive(query)
        
        # Use AI model for deeper analysis if available
        ai_analysis = {}
        if self.ai_model:
            ai_analysis = self._ai_query_analysis(query)
        
        return {
            "query_type": query_type,
            "information_type": info_type,
            "key_concepts": key_concepts,
            "time_sensitive": time_sensitive,
            "ai_analysis": ai_analysis
        }
    
    def _determine_query_type(self, query: str) -> str:
        """
        Determine the type of query.
        
        Args:
            query: User's research query
            
        Returns:
            Query type (factual, exploratory, news, opinion, how-to)
        """
        query_lower = query.lower()
        
        # Check for factual queries
        if any(term in query_lower for term in ["what is", "who is", "when did", "where is", "define", "meaning of"]):
            return "factual"
        
        # Check for news queries
        if any(term in query_lower for term in ["latest", "recent", "news", "update", "current", "today", "yesterday"]):
            return "news"
        
        # Check for how-to queries
        if any(term in query_lower for term in ["how to", "steps to", "guide", "tutorial", "instructions"]):
            return "how-to"
        
        # Check for opinion queries
        if any(term in query_lower for term in ["opinion", "review", "best", "worst", "pros and cons", "should i"]):
            return "opinion"
        
        # Default to exploratory
        return "exploratory"
    
    def _identify_information_type(self, query: str) -> str:
        """
        Identify the type of information needed.
        
        Args:
            query: User's research query
            
        Returns:
            Information type (facts, opinions, news, historical, technical)
        """
        query_lower = query.lower()
        
        # Check for news information
        if any(term in query_lower for term in ["latest", "recent", "news", "update", "current"]):
            return "news"
        
        # Check for historical information
        if any(term in query_lower for term in ["history", "historical", "past", "origin", "evolution"]):
            return "historical"
        
        # Check for technical information
        if any(term in query_lower for term in ["technical", "technology", "code", "programming", "engineering"]):
            return "technical"
        
        # Check for opinion information
        if any(term in query_lower for term in ["opinion", "review", "best", "worst", "pros and cons"]):
            return "opinions"
        
        # Default to facts
        return "facts"
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """
        Extract key concepts from the query.
        
        Args:
            query: User's research query
            
        Returns:
            List of key concepts
        """
        # Remove common question words and stop words
        stop_words = [
            "what", "who", "when", "where", "why", "how", "is", "are", "was", "were",
            "do", "does", "did", "can", "could", "should", "would", "the", "a", "an",
            "in", "on", "at", "to", "for", "with", "about", "of", "and", "or"
        ]
        
        # Tokenize the query
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Extract phrases (2-3 word combinations)
        phrases = []
        for i in range(len(words) - 1):
            if words[i] not in stop_words or words[i+1] not in stop_words:
                phrases.append(f"{words[i]} {words[i+1]}")
        
        for i in range(len(words) - 2):
            if words[i] not in stop_words or words[i+1] not in stop_words or words[i+2] not in stop_words:
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # Combine individual words and phrases
        concepts = filtered_words + phrases
        
        # Remove duplicates and sort by length (longer phrases first)
        unique_concepts = list(set(concepts))
        unique_concepts.sort(key=len, reverse=True)
        
        # Return top concepts (limit to 5)
        return unique_concepts[:5]
    
    def _is_time_sensitive(self, query: str) -> bool:
        """
        Check if the query is time-sensitive.
        
        Args:
            query: User's research query
            
        Returns:
            Boolean indicating if the query is time-sensitive
        """
        time_indicators = [
            "latest", "recent", "current", "today", "yesterday", "this week",
            "this month", "this year", "now", "update", "news"
        ]
        
        query_lower = query.lower()
        
        return any(indicator in query_lower for indicator in time_indicators)
    
    def _ai_query_analysis(self, query: str) -> Dict[str, Any]:
        """
        Use AI model to analyze the query.
        
        Args:
            query: User's research query
            
        Returns:
            Dictionary containing AI analysis results
        """
        if not self.ai_model:
            return {}
        
        try:
            # This is a placeholder for actual AI model integration
            # In a real implementation, this would call the AI model API
            
            # Example prompt for the AI model
            prompt = f"""
            Analyze the following research query:
            
            "{query}"
            
            Please provide:
            1. The main intent of the query
            2. Key concepts that need to be researched
            3. The type of information needed (facts, opinions, news, etc.)
            4. Whether time-sensitive information is required
            5. Suggested search strategies
            """
            
            # Mock AI response for demonstration
            ai_response = {
                "intent": "The user wants to understand the concept and applications of X",
                "key_concepts": ["concept1", "concept2", "concept3"],
                "information_type": "technical facts and practical applications",
                "time_sensitive": False,
                "search_strategies": [
                    "Look for authoritative technical resources",
                    "Find recent applications and case studies",
                    "Check for tutorials and guides"
                ]
            }
            
            return ai_response
        except Exception as e:
            print(f"AI query analysis error: {e}")
            return {}
    
    def _generate_search_queries(self, original_query: str, query_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate effective search queries based on the original query and analysis.
        
        Args:
            original_query: User's original research query
            query_analysis: Analysis of the query
            
        Returns:
            List of search queries
        """
        search_queries = [original_query]  # Start with the original query
        
        # Add queries based on key concepts
        key_concepts = query_analysis.get("key_concepts", [])
        query_type = query_analysis.get("query_type", "exploratory")
        info_type = query_analysis.get("information_type", "facts")
        
        # Generate concept-based queries
        for concept in key_concepts[:3]:  # Use top 3 concepts
            if query_type == "factual":
                search_queries.append(f"what is {concept}")
                search_queries.append(f"{concept} definition explanation")
            
            elif query_type == "news":
                search_queries.append(f"{concept} latest news")
                search_queries.append(f"{concept} recent developments")
            
            elif query_type == "how-to":
                search_queries.append(f"how to {concept} guide")
                search_queries.append(f"{concept} tutorial steps")
            
            elif query_type == "opinion":
                search_queries.append(f"{concept} review analysis")
                search_queries.append(f"{concept} pros and cons")
            
            else:  # exploratory
                search_queries.append(f"{concept} comprehensive guide")
                search_queries.append(f"{concept} explained in detail")
        
        # Add time-sensitive modifiers if needed
        if query_analysis.get("time_sensitive", False):
            time_queries = []
            for query in search_queries[:2]:  # Apply to top 2 queries
                time_queries.append(f"{query} latest")
                time_queries.append(f"{query} recent")
            search_queries.extend(time_queries)
        
        # Remove duplicates and limit to 5 queries
        unique_queries = list(dict.fromkeys(search_queries))
        return unique_queries[:5]
    
    def _perform_search(self, search_query: str, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform web search using the search query.
        
        Args:
            search_query: Search query
            query_analysis: Analysis of the original query
            
        Returns:
            List of search results
        """
        results = []
        
        # Determine if we should use news search
        use_news = query_analysis.get("query_type") == "news" or query_analysis.get("information_type") == "news"
        
        if use_news:
            # Perform news search
            news_results = self.news_aggregator.search_news(
                query=search_query,
                days=7,
                max_results=10
            )
            
            if "articles" in news_results:
                for article in news_results["articles"]:
                    results.append({
                        "title": article.get("title", ""),
                        "link": article.get("url", ""),
                        "snippet": article.get("description", ""),
                        "source": "news",
                        "date": article.get("publishedAt", "")
                    })
        
        # Perform web search
        search_results = self.search_tool.search(
            query=search_query,
            num_results=10,
            search_type="web"
        )
        
        if "organic_results" in search_results:
            for result in search_results["organic_results"]:
                results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "source": "web",
                    "position": result.get("position", 0)
                })
        
        return results
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate search results.
        
        Args:
            results: List of search results
            
        Returns:
            List of deduplicated search results
        """
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get("link", "")
            
            # Normalize URL for comparison
            parsed_url = urlparse(url)
            normalized_url = f"{parsed_url.netloc}{parsed_url.path}"
            
            if normalized_url and normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[Dict[str, Any]], query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank search results by relevance.
        
        Args:
            results: List of search results
            query_analysis: Analysis of the original query
            
        Returns:
            List of ranked search results
        """
        # Calculate relevance score for each result
        scored_results = []
        
        for result in results:
            score = 0
            
            # Base score from search position (if available)
            position = result.get("position", 0)
            if position > 0:
                score += max(0, 10 - position) * 0.5
            
            # Check title and snippet for key concepts
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            
            for concept in query_analysis.get("key_concepts", []):
                concept_lower = concept.lower()
                if concept_lower in title:
                    score += 3
                if concept_lower in snippet:
                    score += 2
            
            # Boost news sources for time-sensitive queries
            if query_analysis.get("time_sensitive", False) and result.get("source") == "news":
                score += 5
            
            # Add the score to the result
            result["relevance_score"] = score
            scored_results.append(result)
        
        # Sort by relevance score (descending)
        scored_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return scored_results
    
    def _extract_and_analyze_content(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Extract and analyze content from a search result.
        
        Args:
            result: Search result
            query: Original query
            
        Returns:
            Dictionary containing extracted and analyzed content
        """
        url = result.get("link", "")
        if not url:
            return None
        
        # Scrape the content
        scraped_data = self.scraper.scrape(url)
        
        if not scraped_data.get("success", False):
            print(f"Failed to scrape {url}: {scraped_data.get('error', 'Unknown error')}")
            return None
        
        content = scraped_data.get("content", {})
        
        # Analyze the content
        analysis = self.content_analyzer.analyze(content, query)
        
        # Combine result, content, and analysis
        return {
            "url": url,
            "title": content.get("title", result.get("title", "")),
            "main_content": content.get("main_content", ""),
            "relevance_score": analysis.get("relevance_score", 0),
            "information_density": analysis.get("information_density", 0),
            "recency": analysis.get("recency", {}),
            "reliability": analysis.get("reliability", 0),
            "key_sentences": analysis.get("key_sentences", []),
            "ai_analysis": analysis.get("ai_analysis", {})
        }
    
    def _synthesize_information(self, analyzed_sources: List[Dict[str, Any]], query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize information from multiple sources.
        
        Args:
            analyzed_sources: List of analyzed sources
            query: Original query
            query_analysis: Analysis of the query
            
        Returns:
            Dictionary containing synthesized information
        """
        if not analyzed_sources:
            return {
                "key_information": [],
                "contradictions": [],
                "summary": "No information could be found for this query."
            }
        
        # Extract key information from all sources
        all_key_sentences = []
        for source in analyzed_sources:
            for sentence in source.get("key_sentences", []):
                all_key_sentences.append({
                    "sentence": sentence,
                    "source_url": source["url"],
                    "source_title": source["title"],
                    "relevance_score": source["relevance_score"],
                    "reliability": source["reliability"]
                })
        
        # Sort key sentences by source relevance and reliability
        all_key_sentences.sort(key=lambda x: (x["relevance_score"] + x["reliability"]), reverse=True)
        
        # Group similar information
        grouped_information = self._group_similar_information(all_key_sentences)
        
        # Identify contradictions
        contradictions = self._identify_contradictions(grouped_information)
        
        # Generate summary
        summary = self._generate_summary(grouped_information, contradictions, query)
        
        return {
            "key_information": grouped_information,
            "contradictions": contradictions,
            "summary": summary
        }
    
    def _group_similar_information(self, key_sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group similar information from different sources.
        
        Args:
            key_sentences: List of key sentences with source information
            
        Returns:
            List of grouped information
        """
        # This is a simplified implementation
        # In a real implementation, this would use more sophisticated text similarity
        
        groups = []
        
        for sentence_info in key_sentences:
            sentence = sentence_info["sentence"].lower()
            
            # Check if this sentence is similar to any existing group
            found_group = False
            
            for group in groups:
                group_sentence = group["sentences"][0]["sentence"].lower()
                
                # Simple similarity check (can be improved with NLP techniques)
                # Check if 50% of words in shorter sentence are in longer sentence
                words1 = set(re.findall(r'\b\w+\b', sentence))
                words2 = set(re.findall(r'\b\w+\b', group_sentence))
                
                if len(words1) == 0 or len(words2) == 0:
                    continue
                
                # Calculate overlap
                overlap = len(words1.intersection(words2))
                similarity = overlap / min(len(words1), len(words2))
                
                if similarity > 0.5:
                    group["sentences"].append(sentence_info)
                    found_group = True
                    break
            
            # If no similar group found, create a new one
            if not found_group:
                groups.append({
                    "sentences": [sentence_info],
                    "sources": [sentence_info["source_url"]],
                    "reliability": sentence_info["reliability"]
                })
            else:
                # Update group sources
                group["sources"] = list(set(group["sources"] + [sentence_info["source_url"]]))
                # Update reliability (average)
                group["reliability"] = sum(s["reliability"] for s in group["sentences"]) / len(group["sentences"])
        
        # Sort groups by number of sources and reliability
        groups.sort(key=lambda x: (len(x["sources"]), x["reliability"]), reverse=True)
        
        return groups
    
    def _identify_contradictions(self, grouped_information: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify contradictory information between sources.
        
        Args:
            grouped_information: Grouped information from different sources
            
        Returns:
            List of contradictions
        """
        # This is a simplified implementation
        # In a real implementation, this would use more sophisticated contradiction detection
        
        contradictions = []
        
        # Look for potential contradictions between groups
        for i in range(len(grouped_information)):
            for j in range(i + 1, len(grouped_information)):
                group1 = grouped_information[i]
                group2 = grouped_information[j]
                
                # Get representative sentences
                sentence1 = group1["sentences"][0]["sentence"].lower()
                sentence2 = group2["sentences"][0]["sentence"].lower()
                
                # Check for contradiction indicators
                contradiction_indicators = [
                    ("increase", "decrease"),
                    ("higher", "lower"),
                    ("more", "less"),
                    ("positive", "negative"),
                    ("good", "bad"),
                    ("yes", "no"),
                    ("true", "false"),
                    ("agree", "disagree"),
                    ("support", "oppose"),
                    ("confirm", "deny")
                ]
                
                for word1, word2 in contradiction_indicators:
                    if (word1 in sentence1 and word2 in sentence2) or (word2 in sentence1 and word1 in sentence2):
                        # Potential contradiction found
                        contradictions.append({
                            "group1": group1,
                            "group2": group2,
                            "contradiction_type": f"{word1} vs {word2}"
                        })
                        break
        
        return contradictions
    
    def _generate_summary(self, grouped_information: List[Dict[str, Any]], contradictions: List[Dict[str, Any]], query: str) -> str:
        """
        Generate a summary of the synthesized information.
        
        Args:
            grouped_information: Grouped information from different sources
            contradictions: List of contradictions
            query: Original query
            
        Returns:
            Summary text
        """
        if not grouped_information:
            return "No relevant information could be found for this query."
        
        # Use AI model for summary generation if available
        if self.ai_model:
            return self._ai_generate_summary(grouped_information, contradictions, query)
        
        # Manual summary generation
        summary_parts = []
        
        # Add introduction
        summary_parts.append(f"Based on research from {len(set(sum([g['sources'] for g in grouped_information], [])))} sources:")
        
        # Add key information
        summary_parts.append("\nKey findings:")
        
        for i, group in enumerate(grouped_information[:5]):  # Top 5 groups
            # Use the most reliable sentence in the group
            best_sentence = max(group["sentences"], key=lambda x: x["reliability"])
            summary_parts.append(f"- {best_sentence['sentence']} (Found in {len(group['sources'])} sources)")
        
        # Add contradictions if any
        if contradictions:
            summary_parts.append("\nContradictory information:")
            
            for i, contradiction in enumerate(contradictions[:3]):  # Top 3 contradictions
                group1 = contradiction["group1"]
                group2 = contradiction["group2"]
                
                summary_parts.append(f"- Contradiction: {group1['sentences'][0]['sentence']} vs {group2['sentences'][0]['sentence']}")
        
        return "\n".join(summary_parts)
    
    def _ai_generate_summary(self, grouped_information: List[Dict[str, Any]], contradictions: List[Dict[str, Any]], query: str) -> str:
        """
        Generate a summary of the synthesized information using AI.
        
        Args:
            grouped_information: Grouped information from different sources
            contradictions: List of contradictions
            query: Original query
            
        Returns:
            Summary text
        """
        if not self.ai_model:
            return self._generate_summary(grouped_information, contradictions, query)
        
        try:
            # Prepare input for the AI model
            input_text = f"Query: {query}\n\nKey Information:\n"
            
            for i, group in enumerate(grouped_information[:10]):  # Top 10 groups
                best_sentence = max(group["sentences"], key=lambda x: x["reliability"])
                input_text += f"{i+1}. {best_sentence['sentence']} (Found in {len(group['sources'])} sources, reliability: {group['reliability']:.2f})\n"
            
            if contradictions:
                input_text += "\nContradictions:\n"
                for i, contradiction in enumerate(contradictions[:5]):  # Top 5 contradictions
                    group1 = contradiction["group1"]
                    group2 = contradiction["group2"]
                    input_text += f"{i+1}. {group1['sentences'][0]['sentence']} vs {group2['sentences'][0]['sentence']}\n"
            
            # Example prompt for the AI model
            prompt = f"""
            Based on the following information gathered from web research, please provide a comprehensive summary that answers the query.
            
            {input_text}
            
            Please:
            1. Synthesize the information into a coherent answer
            2. Address any contradictions by evaluating source reliability
            3. Indicate confidence levels for different parts of the answer
            4. Organize the information logically
            5. Keep the summary concise but comprehensive
            """
            
            # Mock AI response for demonstration
            ai_response = f"Based on the research, the answer to '{query}' is as follows:\n\n"
            ai_response += "The analysis of multiple sources indicates that... [AI-generated summary would go here]"
            
            return ai_response
        except Exception as e:
            print(f"AI summary generation error: {e}")
            # Fall back to manual summary
            return self._generate_summary(grouped_information, contradictions, query)
    
    def _generate_answer(self, synthesis: Dict[str, Any], query: str) -> str:
        """
        Generate the final answer to the user's query.
        
        Args:
            synthesis: Synthesized information
            query: Original query
            
        Returns:
            Final answer text
        """
        # Start with the summary
        answer = synthesis.get("summary", "")
        
        # Add source citations
        sources = []
        for group in synthesis.get("key_information", []):
            for source_url in group.get("sources", []):
                if source_url not in sources:
                    sources.append(source_url)
        
        if sources:
            answer += "\n\nSources:\n"
            for i, source in enumerate(sources[:5]):  # Limit to top 5 sources
                answer += f"{i+1}. {source}\n"
        
        return answer
