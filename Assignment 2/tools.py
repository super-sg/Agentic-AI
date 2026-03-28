"""
Tools for the Autonomous Research Agent.
Provides Web Search (DuckDuckGo) and Knowledge (Wikipedia) tools.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


def get_search_tool():
    """
    Create and return a DuckDuckGo web search tool.
    Free to use, no API key required.
    """
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
    search_tool = DuckDuckGoSearchRun(
        api_wrapper=wrapper,
        name="web_search",
        description=(
            "Search the web for current information about a topic. "
            "Use this tool to find recent news, articles, research papers, "
            "and general information. Input should be a search query string."
        ),
    )
    return search_tool


def get_wikipedia_tool():
    """
    Create and return a Wikipedia knowledge tool.
    Retrieves structured information from Wikipedia articles.
    """
    wrapper = WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=4000,
    )
    wiki_tool = WikipediaQueryRun(
        api_wrapper=wrapper,
        name="wikipedia",
        description=(
            "Search Wikipedia for detailed, structured information about a topic. "
            "Use this for background knowledge, definitions, historical context, "
            "and well-established facts. Input should be a search query string."
        ),
    )
    return wiki_tool


def get_all_tools():
    """Return a list of all available tools."""
    return [get_search_tool(), get_wikipedia_tool()]
