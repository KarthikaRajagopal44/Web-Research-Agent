import os
import json
import time
from typing import Dict, Any, Optional
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs

# Import the WebResearchAgent
from web_research_agent import WebResearchAgent

# Import AI model integrations
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class AIModelWrapper:
    """
    Wrapper for AI model integrations.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the AI model wrapper.
        
        Args:
            model_name: Name of the AI model to use
            api_key: API key for the AI service
        """
        self.model_name = model_name.lower()
        self.api_key = api_key
        self.client = None
        
        # Initialize the appropriate client
        if "gpt" in self.model_name and OPENAI_AVAILABLE:
            self.provider = "openai"
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        
        elif "gemini" in self.model_name and GEMINI_AVAILABLE:
            self.provider = "gemini"
            genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
            self.client = genai
        
        elif "claude" in self.model_name and ANTHROPIC_AVAILABLE:
            self.provider = "anthropic"
            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        
        else:
            self.provider = "mock"
            print(f"Warning: Model {model_name} not available or required libraries not installed. Using mock implementation.")
    
    def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate text using the AI model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API error: {e}")
                return self._mock_generate_text(prompt)
        
        elif self.provider == "gemini":
            try:
                model = self.client.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini API error: {e}")
                return self._mock_generate_text(prompt)
        
        elif self.provider == "anthropic":
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                print(f"Anthropic API error: {e}")
                return self._mock_generate_text(prompt)
        
        else:
            return self._mock_generate_text(prompt)
    
    def _mock_generate_text(self, prompt: str) -> str:
        """
        Mock implementation of text generation.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Mock generated text
        """
        return f"This is a mock response to the prompt: {prompt[:50]}..."

def run_research(query: str, model: str = "gpt-4o", depth: int = 2, max_sources: int = 5, 
                search_api_key: str = None, news_api_key: str = None, ai_api_key: str = None, 
                use_mock: bool = False) -> Dict[str, Any]:
    """
    Run web research on the given query.
    
    Args:
        query: Research query
        model: AI model to use
        depth: Research depth (1-3)
        max_sources: Maximum number of sources to analyze
        search_api_key: API key for search service
        news_api_key: API key for news service
        ai_api_key: API key for AI service
        use_mock: Whether to use mock implementations
        
    Returns:
        Dictionary containing research results
    """
    # Initialize AI model
    ai_model = AIModelWrapper(model, ai_api_key)
    
    # Initialize web research agent
    agent = WebResearchAgent(
        ai_model=ai_model,
        search_api_key=search_api_key,
        news_api_key=news_api_key,
        use_mock=use_mock
    )
    
    # Perform research
    print(f"Researching: {query}")
    start_time = time.time()
    
    results = agent.research(
        query=query,
        depth=depth,
        max_sources=max_sources
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Research completed in {duration:.2f} seconds")
    
    # Add duration to results
    results["duration"] = duration
    
    return results

# For command-line usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Research Agent")
    parser.add_argument("query", type=str, help="Research query")
    parser.add_argument("--model", type=str, default="gpt-4o", help="AI model to use")
    parser.add_argument("--depth", type=int, default=2, help="Research depth (1-3)")
    parser.add_argument("--max-sources", type=int, default=5, help="Maximum number of sources to analyze")
    parser.add_argument("--output", type=str, default="research_results.json", help="Output file path")
    parser.add_argument("--search-api-key", type=str, help="API key for search service")
    parser.add_argument("--news-api-key", type=str, help="API key for news service")
    parser.add_argument("--ai-api-key", type=str, help="API key for AI service")
    parser.add_argument("--use-mock", action="store_true", help="Use mock implementations instead of real APIs")
    
    args = parser.parse_args()
    
    # Run research
    results = run_research(
        query=args.query,
        model=args.model,
        depth=args.depth,
        max_sources=args.max_sources,
        search_api_key=args.search_api_key,
        news_api_key=args.news_api_key,
        ai_api_key=args.ai_api_key,
        use_mock=args.use_mock
    )
    
    # Print answer
    print("\nAnswer:")
    print(results["answer"])
    
    # Print sources
    print("\nSources:")
    for i, source in enumerate(results["sources"]):
        print(f"{i+1}. {source['title']} - {source['url']}")
    
    # Save results to file
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")

# For Vercel serverless function
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Web Research Agent API. Use POST to submit a query.')
        return
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            # Try to parse as JSON
            data = json.loads(post_data)
        except json.JSONDecodeError:
            # If not JSON, try to parse as form data
            data = parse_qs(post_data)
            # Convert lists to single values
            data = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in data.items()}
        
        # Extract parameters
        query = data.get('query')
        model = data.get('model', 'gpt-4o')
        depth = int(data.get('depth', 2))
        max_sources = int(data.get('max_sources', 5))
        use_mock = data.get('use_mock', 'false').lower() == 'true'
        
        # Get API keys from environment or request
        search_api_key = data.get('search_api_key') or os.environ.get('SEARCH_API_KEY')
        news_api_key = data.get('news_api_key') or os.environ.get('NEWS_API_KEY')
        ai_api_key = data.get('ai_api_key') or os.environ.get('AI_API_KEY')
        
        if not query:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Missing query parameter"}).encode())
            return
        
        # Run research
        try:
            results = run_research(
                query=query,
                model=model,
                depth=depth,
                max_sources=max_sources,
                search_api_key=search_api_key,
                news_api_key=news_api_key,
                ai_api_key=ai_api_key,
                use_mock=use_mock
            )
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(results).encode())
        
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
