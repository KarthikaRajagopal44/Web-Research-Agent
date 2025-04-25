import requests
from bs4 import BeautifulSoup
import time
import random
from typing import Dict, Any, List, Optional
import re
from urllib.parse import urlparse
import os

class WebScraper:
    """
    Tool for scraping content from web pages.
    
    This tool extracts text, structured data, and other relevant information from web pages.
    """
    
    def __init__(self, user_agent: Optional[str] = None, respect_robots: bool = True):
        """
        Initialize the WebScraper.
        
        Args:
            user_agent: User agent string to use for requests
            respect_robots: Whether to respect robots.txt directives
        """
        self.user_agent = user_agent or "WebResearchAgent/1.0"
        self.respect_robots = respect_robots
        self.robots_cache = {}  # Cache for robots.txt content
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
    def scrape(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from the given URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary containing scraped content
        """
        if self.respect_robots and not self._can_fetch(url):
            return {
                "url": url,
                "success": False,
                "error": "Access disallowed by robots.txt",
                "content": {},
            }
        
        try:
            # Add a small delay to be respectful to servers
            time.sleep(random.uniform(1.0, 2.0))
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract content
            content = self._extract_content(soup, url)
            
            return {
                "url": url,
                "success": True,
                "status_code": response.status_code,
                "content": content,
            }
        except requests.RequestException as e:
            return {
                "url": url,
                "success": False,
                "error": str(e),
                "content": {},
            }
    
    def _extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract content from the parsed HTML.
        
        Args:
            soup: BeautifulSoup object of the parsed HTML
            url: Original URL
            
        Returns:
            Dictionary containing extracted content
        """
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
        
        # Get the page title
        title = soup.title.string if soup.title else ""
        
        # Extract main content
        main_content = ""
        
        # Try to find the main content area
        main_tags = soup.find_all(["main", "article", "div", "section"], class_=lambda c: c and any(x in str(c).lower() for x in ["content", "main", "article", "body"]))
        
        if main_tags:
            # Use the largest content area
            main_tag = max(main_tags, key=lambda tag: len(tag.get_text()))
            main_content = main_tag.get_text(separator="\n", strip=True)
        else:
            # If no main content area found, use the body
            main_content = soup.body.get_text(separator="\n", strip=True) if soup.body else ""
        
        # Extract headings
        headings = []
        for h in soup.find_all(["h1", "h2", "h3"]):
            headings.append({
                "level": int(h.name[1]),
                "text": h.get_text(strip=True)
            })
        
        # Extract tables
        tables = []
        for table in soup.find_all("table"):
            table_data = []
            rows = table.find_all("tr")
            
            # Extract headers
            headers = []
            header_row = table.find("thead")
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
            elif rows and rows[0].find_all("th"):
                headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
                rows = rows[1:]  # Skip the header row
            
            # Extract data rows
            for row in rows:
                cells = row.find_all(["td", "th"])
                if cells:
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    table_data.append(row_data)
            
            tables.append({
                "headers": headers,
                "data": table_data
            })
        
        # Extract lists
        lists = []
        for list_tag in soup.find_all(["ul", "ol"]):
            list_items = [li.get_text(strip=True) for li in list_tag.find_all("li")]
            lists.append({
                "type": list_tag.name,
                "items": list_items
            })
        
        # Extract metadata
        metadata = {}
        for meta in soup.find_all("meta"):
            name = meta.get("name") or meta.get("property")
            content = meta.get("content")
            if name and content:
                metadata[name] = content
        
        return {
            "title": title,
            "main_content": main_content,
            "headings": headings,
            "tables": tables,
            "lists": lists,
            "metadata": metadata,
            "full_text": soup.get_text(separator="\n", strip=True),
        }
    
    def _can_fetch(self, url: str) -> bool:
        """
        Check if the URL can be fetched according to robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            Boolean indicating if the URL can be fetched
        """
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = f"{base_url}/robots.txt"
        
        # Check cache first
        if robots_url in self.robots_cache:
            return self._check_robots_rules(self.robots_cache[robots_url], parsed_url.path)
        
        try:
            response = requests.get(robots_url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                self.robots_cache[robots_url] = response.text
                return self._check_robots_rules(response.text, parsed_url.path)
            else:
                # If robots.txt doesn't exist or can't be accessed, assume allowed
                return True
        except requests.RequestException:
            # If there's an error accessing robots.txt, assume allowed
            return True
    
    def _check_robots_rules(self, robots_txt: str, path: str) -> bool:
        """
        Check if the path is allowed according to robots.txt rules.
        
        Args:
            robots_txt: Content of robots.txt
            path: Path to check
            
        Returns:
            Boolean indicating if the path is allowed
        """
        # Simple robots.txt parser
        user_agent_sections = re.split(r"User-agent:", robots_txt)
        
        # Check rules for our user agent and for *
        relevant_sections = []
        for section in user_agent_sections:
            if not section.strip():
                continue
            
            lines = section.strip().split("\n")
            agent = lines[0].strip()
            
            if agent == "*" or self.user_agent in agent:
                relevant_sections.append("\n".join(lines[1:]))
        
        # Check if path is disallowed in any relevant section
        for section in relevant_sections:
            for line in section.split("\n"):
                if line.strip().startswith("Disallow:"):
                    disallow_path = line.replace("Disallow:", "").strip()
                    
                    if disallow_path and path.startswith(disallow_path):
                        return False
        
        return True
