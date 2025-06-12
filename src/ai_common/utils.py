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
