from pydantic import SecretStr
from tavily import AsyncTavilyClient

from .enums import TavilySearchCategory, TavilySearchDepth
from .utils import tavily_search_async, deduplicate_sources


class WebSearch:
    """
    A comprehensive web search interface that leverages Tavily's search API for information retrieval.
    
    This class provides a high-level interface for performing web searches using the Tavily search service,
    with built-in support for multiple search queries, result deduplication, and time-based filtering.
    The WebSearch class is designed to facilitate efficient information gathering for research workflows,
    content analysis, and knowledge extraction tasks.
    
    Key Features:
    - Asynchronous search operations for improved performance
    - Support for multiple concurrent search queries
    - Automatic deduplication of search results across queries
    - Time-based filtering to retrieve recent information
    - Configurable result limits per query
    - Support for different search categories (news, general)
    
    The class handles the complexity of managing multiple search queries, coordinating asynchronous
    operations, and consolidating results into a unified response format. It abstracts away the
    low-level details of the Tavily API while providing flexible configuration options for
    different use cases.
    
    Typical Use Cases:
    - Research and information gathering workflows
    - Content analysis and summarization tasks
    - Real-time news monitoring and analysis
    - Market research and competitive intelligence
    - Academic research and literature review
    - Fact-checking and verification processes
    
    Example:
        >>> import asyncio
        >>> from ai_common.web_search import WebSearch
        >>> 
        >>> # Initialize the web search client
        >>> web_search = WebSearch(api_key="your_tavily_api_key")
        >>> 
        >>> # Perform a search with multiple queries
        >>> queries = [
        ...     "artificial intelligence trends 2024",
        ...     "machine learning applications healthcare",
        ...     "AI ethics regulatory frameworks"
        ... ]
        >>> 
        >>> # Execute the search
        >>> results = 'await' web_search.search(
        ...     search_queries=queries,
        ...     search_category="general",
        ...     number_of_days_back=30,
        ...     max_results_per_query=5
        ... )
        >>> 
        >>> # Process the consolidated results
        >>> print(f"Found {len(results)} unique sources")
        >>> for source in results:
        ...     print(f"Title: {source['title']}")
        ...     print(f"URL: {source['url']}")
    
    Attributes:
        client (AsyncTavilyClient): The underlying Tavily API client used for search operations.
                                  Configured with the provided API key for authenticated requests.
    
    Note:
        This class requires a valid Tavily API key for operation. All search operations are
        asynchronous and should be awaited when called. The class automatically handles
        result deduplication to prevent duplicate sources across multiple queries.
    
    Dependencies:
        - tavily: For the AsyncTavilyClient and search functionality
        - ai_common.base: For TavilySearchCategory type definitions
        - ai_common.utils: For tavily_search_async and deduplicate_sources utilities
    
    Thread Safety:
        The AsyncTavilyClient is designed to be thread-safe, but it's recommended to use
        this class within a single asyncio event loop context for optimal performance.
    """
    def __init__(self, api_key: SecretStr):
        self.client = AsyncTavilyClient(api_key=api_key.get_secret_value())

    async def search(self,
                     search_queries: list[str],
                     search_category: TavilySearchCategory,
                     search_depth: TavilySearchDepth,
                     chunks_per_source: int,
                     number_of_days_back: int,
                     max_results_per_query: int,
                     include_images: bool,
                     include_image_descriptions: bool,
                     include_favicon: bool) -> dict:

        search_docs = await tavily_search_async(
            client=self.client,
            search_queries=search_queries,
            search_category=search_category,
            search_depth=search_depth,
            chunks_per_source=chunks_per_source,
            number_of_days_back=number_of_days_back,
            max_results=max_results_per_query,
            include_images=include_images,
            include_image_descriptions=include_image_descriptions,
            include_favicon=include_favicon,
        )

        unique_sources = deduplicate_sources(search_response=search_docs)
        return unique_sources
