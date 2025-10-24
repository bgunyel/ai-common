import asyncio
import datetime
from io import BytesIO
import importlib

from PIL import Image
from tavily import AsyncTavilyClient
from langchain_core.runnables import RunnableConfig

from .base import CfgBase
from .enums import TavilySearchCategory, TavilySearchDepth


def get_config_from_runnable(configuration_module_prefix: str, config: RunnableConfig) -> CfgBase:
    module = importlib.import_module(name=f'{configuration_module_prefix}')
    class_ = getattr(module, 'Configuration')
    configurable = class_.from_runnable(runnable=config)
    return configurable


def get_flow_chart(rag_model):
    img_bytes = BytesIO(rag_model.graph.get_graph(xray=True).draw_mermaid_png())
    img = Image.open(img_bytes).convert("RGB")
    return img


async def tavily_search_async(client: AsyncTavilyClient,
                              search_queries: list[str],
                              search_category: TavilySearchCategory,
                              search_depth: TavilySearchDepth,
                              chunks_per_source: int,
                              number_of_days_back: int,
                              max_results: int,
                              include_images: bool,
                              include_image_descriptions: bool,
                              include_favicon: bool):
    """
    Perform concurrent web searches using the Tavily API with comprehensive configuration options.
    
    This asynchronous function executes multiple search queries concurrently using the Tavily search API,
    providing extensive control over search behavior, content retrieval, and result formatting. It is
    designed to efficiently handle bulk search operations while maintaining flexibility in search
    configuration and result customization.
    
    The function leverages asyncio.gather to execute all searches concurrently, significantly improving
    performance when processing multiple queries. It supports both news and general search categories,
    with time-based filtering for news searches and comprehensive content retrieval options.
    
    Search Configuration:
    - Supports both 'news' and 'general' search categories
    - Configurable search depth for basic or advanced search algorithms
    - Time-based filtering for news searches (days back from current date)
    - Customizable result limits per query
    - Raw content inclusion for comprehensive text analysis
    - Image and favicon retrieval options
    
    Content Retrieval Options:
    - Raw content extraction for full-text analysis
    - Chunked content processing for better organization
    - Image inclusion with optional descriptions
    - Favicon retrieval for source branding
    - Flexible content structuring based on chunks per source
    
    Args:
        client (AsyncTavilyClient): An authenticated Tavily API client instance for making search requests.
        search_queries (list[str]): List of search query strings to execute concurrently.
                                   Each query will be processed as an independent search operation.
        search_category (TavilySearchCategory): The type of search to perform, either 'news' or 'general'.
                                              Determines the search algorithm and result filtering applied.
        search_depth (TavilySearchDepth): The depth of search to perform, either 'basic' or 'advanced'.
                                        Advanced search provides more comprehensive results but takes longer.
        chunks_per_source (int): Number of content chunks to extract per source.
                               Controls the granularity of content segmentation for better processing.
        number_of_days_back (int): Number of days to look back for news articles.
                                 Only applicable when search_category is 'news'.
                                 Filters results to include only recent articles within the specified timeframe.
        max_results (int): The maximum number of search results to return per query.
                          Controls the volume of results and API usage.
        include_images (bool): Whether to include images in the search results.
                             Adds visual content to results for multimedia analysis.
        include_image_descriptions (bool): Whether to include AI-generated descriptions for images.
                                         Provides textual context for visual content.
        include_favicon (bool): Whether to include website favicons in the results.
                              Adds branding information for source identification.
    
    Returns:
        List[dict]: A list of search result dictionaries, one per input query, preserving the order
                   of input queries. Each dictionary contains the complete search response from the
                   Tavily API including:
                   - results: List of individual search result items
                   - query: The original search query
                   - response_time: Time taken to process the search
                   - Additional metadata based on configuration options
    
    Example:
        >>> import asyncio
        >>> from tavily import AsyncTavilyClient
        >>> from ai_common.utils import tavily_search_async
        >>> 
        >>> # Initialize client
        >>> client_ = AsyncTavilyClient(api_key="your_api_key")
        >>> 
        >>> # Define search queries
        >>> queries = [
        ...     "artificial intelligence trends 2024",
        ...     "machine learning healthcare applications",
        ...     "quantum computing recent developments"
        ... ]
        >>> 
        >>> # Execute concurrent searches
        >>> results = 'await' tavily_search_async(
        ...     client=client_,
        ...     search_queries=queries,
        ...     search_category="general",
        ...     search_depth="advanced",
        ...     chunks_per_source=3,
        ...     number_of_days_back=30,
        ...     max_results=5,
        ...     include_images=True,
        ...     include_image_descriptions=True,
        ...     include_favicon=True
        ... )
        >>> 
        >>> # Process results
        >>> for i, result in enumerate(results):
        ...     print(f"Query {i+1}: {queries[i]}")
        ...     print(f"Found {len(result['results'])} results")
    
    Performance Characteristics:
        - Concurrent execution using asyncio.gather for optimal performance
        - Network I/O bound operations benefit from async processing
        - Memory usage scales with number of queries and max_results per query
        - Processing time depends on search_depth and API response times
    
    Error Handling:
        The function relies on the underlying AsyncTavilyClient for error handling.
        Network errors, authentication failures, and API rate limits are propagated
        from the client. It's recommended to implement retry logic and error handling
        at the calling level.
    
    Note:
        - For news searches, the number_of_days_back parameter filters results to recent articles
        - For general searches, the time range is unrestricted regardless of number_of_days_back
        - All searches include raw content by default for comprehensive text analysis
        - The function preserves the order of input queries in the returned results
        - API usage and costs scale with the number of queries and max_results per query
    
    See Also:
        - AsyncTavilyClient: The underlying client used for API communication
        - deduplicate_sources: Function for removing duplicate results across searches
        - format_sources: Function for formatting search results for display
    """

    start_date = datetime.date.today() - datetime.timedelta(days=number_of_days_back)
    start_date_str = start_date.isoformat() if start_date > datetime.date.fromisoformat("1919-05-19") else "1919-05-19"

    kwargs = {
        'max_results': max_results,
        'include_raw_content': True,
        'topic': search_category.value,
        'search_depth': search_depth.value,
        'chunks_per_source': chunks_per_source,
        'include_images': include_images,
        'include_image_descriptions': include_image_descriptions,
        'include_favicon': include_favicon,
        'start_date': start_date_str,
    }

    # Execute all searches concurrently
    search_tasks = [client.search(query=query, **kwargs) for query in search_queries]
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs


def deduplicate_sources(search_response: list[dict]):
    """
    Remove duplicate sources from web search results based on URL uniqueness.
    
    This function processes a list of search responses from the Tavily API and eliminates
    duplicate sources by using URLs as unique identifiers. It handles both single search
    responses (containing a 'results' key) and lists of search results, extracting and
    consolidating all sources into a single deduplicated collection.
    
    The deduplication process is essential when performing multiple concurrent searches
    that may return overlapping results. By removing duplicates, the function ensures
    that downstream processing (such as content analysis or summarization) operates
    on a clean, unique set of sources without redundancy.
    
    The function is designed to be robust and handle different response formats from
    the Tavily API, automatically detecting whether each response contains a 'results'
    key (typical API response format) or is a direct list of search results.
    
    Args:
        search_response (list[dict]): A list of search responses from the Tavily API.
                                    Each response can be either:
                                    - A dictionary with a 'results' key containing a list of sources
                                    - A list of source dictionaries directly
                                    Each source dictionary should contain at least a 'url' key
                                    for deduplication purposes.
    
    Returns:
        dict: A dictionary mapping unique URLs to their corresponding source dictionaries.
              The keys are URLs (str) and values are the complete source dictionaries
              from the original search responses. This format allows for efficient
              lookup and preserves all source metadata while ensuring uniqueness.
              
              Structure: {
                  'https://example.com/article1': {
                      'url': 'https://example.com/article1',
                      'title': 'Article Title',
                      'content': 'Article content...',
                      'raw_content': 'Full article text...',
                      ...
                  },
                  ...
              }
    
    Example:
        >>> # Multiple search responses with overlapping results
        >>> responses = [
        ...     {
        ...         'results': [
        ...             {'url': 'https://example.com/ai-trends', 'title': 'AI Trends 2024'},
        ...             {'url': 'https://example.com/ml-health', 'title': 'ML in Healthcare'}
        ...         ]
        ...     },
        ...     {
        ...         'results': [
        ...             {'url': 'https://example.com/ai-trends', 'title': 'AI Trends 2024'},  # Duplicate
        ...             {'url': 'https://example.com/ai-ethics', 'title': 'AI Ethics Framework'}
        ...         ]
        ...     }
        ... ]
        >>> 
        >>> # Deduplicate the sources
        >>> unique_sources_ = deduplicate_sources(responses)
        >>> print(len(unique_sources_))  # 3 unique sources
        >>> print(list(unique_sources_.keys()))
        ['https://example.com/ai-trends', 'https://example.com/ml-health', 'https://example.com/ai-ethics']
    
    Processing Logic:
        1. Iterate through each search response in the input list
        2. Check if the response is a dictionary with a 'results' key
        3. If yes, extract the results list; otherwise, treat the response as a direct list
        4. Extend the sources list with all extracted results
        5. Create a dictionary using URLs as keys to automatically eliminate duplicates
        6. Return the deduplicated dictionary of unique sources
    
    Note:
        - The function preserves the first occurrence of each URL encountered
        - Source dictionaries must contain a 'url' key for proper deduplication
        - The function is tolerant of different response formats from the Tavily API
        - Memory usage is proportional to the number of unique sources found
        - The returned dictionary format facilitates efficient source lookup operations
    
    See Also:
        - tavily_search_async: Function that generates the search responses processed by this function
        - format_sources: Function that formats the deduplicated sources for display
        - deduplicate_and_format_sources: Convenience function that combines deduplication and formatting
    """
    sources_list = []
    for response in search_response:
        if isinstance(response, dict) and 'results' in response:
            sources_list.extend(response['results'])
        else:
            sources_list.extend(response)

    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source

    return unique_sources


def format_sources(unique_sources: dict,
                   max_tokens_per_source: int = 5000,
                   include_raw_content: bool = True) -> str:

    # Format output
    formatted_text = "Sources:\n\n"
    for i, (url, source) in enumerate(unique_sources.items(), 1):
        formatted_text += f"Source {i}:\n\n"
        formatted_text += f'Title: {source["title"]}\n\n'
        formatted_text += f"URL: {url}\n\n"
        formatted_text += f"Most relevant content from source:\n{source['content']}\n==\n\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                # print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens:\n {raw_content}\n\n"

        formatted_text += '====================================\n\n'

    return formatted_text.strip()


# Modified from: https://github.com/langchain-ai/report-mAIstro/report_masitro.py#L89
def deduplicate_and_format_sources(search_response: list[dict],
                                   max_tokens_per_source: int = 5000,
                                   include_raw_content: bool = True) -> str:
    """
    Takes either a single search response or list of responses from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
        max_tokens_per_source: int
        include_raw_content: Boolean

    Returns:
        str: Formatted string with deduplicated sources
    """

    unique_sources = deduplicate_sources(search_response=search_response)
    out_str = format_sources(unique_sources=unique_sources,
                             max_tokens_per_source=max_tokens_per_source,
                             include_raw_content=include_raw_content)
    return out_str


def strip_thinking_tokens(text: str) -> str:
    """
    NOTE: Original --> https://github.com/langchain-ai/local-deep-researcher/blob/main/src/ollama_deep_researcher/utils.py

    Remove <think> and </think> tags and their content from the text.

    Iteratively removes all occurrences of content enclosed in thinking tokens.

    Args:
        text (str): The text to process

    Returns:
        str: The text with thinking tokens and their content removed
    """
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text
