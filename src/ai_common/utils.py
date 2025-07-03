import asyncio
from io import BytesIO
import importlib

from PIL import Image
from tavily import AsyncTavilyClient
from langchain_core.runnables import RunnableConfig

from .base import TavilySearchCategory, CfgBase


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
                              number_of_days_back: int,
                              max_results: int = 5):
    """
    Performs concurrent web searches using the Tavily API.

    Args:
        client: Async Tavily Client
        search_queries (Queries): List of search queries to process
        search_category (str): Type of search to perform ('news' or 'general')
        number_of_days_back (int): Number of days to look back for news articles (only used when tavily_topic='news')
        max_results (int): The maximum number of search results to return. Default is 5.

    Returns:
        List[dict]: List of search results from Tavily API, one per query

    Note:
        For news searches, each result will include articles from the last `number_of_days_back` days.
        For general searches, the time range is unrestricted.
    """

    kwargs = {
        'max_results': max_results,
        'include_raw_content': True,
        'topic': search_category,
    }
    if search_category == 'news':
        kwargs['days'] = number_of_days_back

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
