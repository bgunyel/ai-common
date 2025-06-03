# ai-common

A Python utility library providing common classes and functions for AI applications.

## Features

- **Multi-LLM Support**: Unified interface for Anthropic, OpenAI, Groq, and Ollama models
- **Web Search Integration**: Async Tavily search with source deduplication and formatting
- **Base Abstractions**: Configuration and graph base classes for AI workflows
- **Utility Functions**: Flow chart generation, thinking token processing, and more
- **Engine Framework**: Core engine abstraction for building AI applications

## Quick Start

Add ai-common as a dependency to your project

```
"ai-common @ git+https://github.com/bgunyel/ai-common.git@main"
```
```python
from ai_common import get_llm, tavily_search_async, LlmServers, WebSearch

# Get an LLM instance
llm = get_llm(LlmServers.OPENAI, model="gpt-4")

# Perform web search
web_search = WebSearch( api_key = "web_search_api_key",
                        search_category = "search_category",
                        number_of_days_back = 3,
                        include_raw_content = True)
unique_sources = web_search.search(search_queries=["AI trends in 2025", "Possible AI trends in 2026"])
```

## Requirements

- Python 3.11+
- Dependencies:
  - langchain-anthropic>=0.3.13
  - langchain-core>=0.3.56
  - langchain-groq>=0.3.2
  - langchain-ollama>=0.3.2
  - langchain-openai>=0.3.15
  - ollama>=0.4.8
  - openai>=1.79.0
  - pillow>=11.2.1
  - tavily-python>=0.7.0
  - tqdm>=4.67.1