# ai-common

A Python utility library providing common classes and functions for AI applications.

## Features

- **Multi-LLM Support**: Unified interface for Anthropic, OpenAI, Groq, and Ollama models
- **Web Search Integration**: Async Tavily search with source deduplication and formatting
- **Graph Components**: Pre-built QueryWriter and WebSearchNode for AI workflows
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
model_params = {
    'model_name': 'gpt-4',
    'openai_api_key': 'your-openai-api-key'
}
llm = get_llm(LlmServers.OPENAI, model_params)

# Perform web search
web_search = WebSearch( api_key = "web_search_api_key",
                        search_category = "search_category",
                        number_of_days_back = 3,
                        include_raw_content = True)
unique_sources = web_search.search(search_queries=["AI trends in 2025", "Possible AI trends in 2026"])
```

## API Documentation

### LLM Configuration

The `get_llm()` function requires different parameters based on the LLM server:

#### OpenAI
```python
model_params = {
    'model_name': 'gpt-4',
    'openai_api_key': 'your-api-key'
}
llm = get_llm(LlmServers.OPENAI, model_params)
```

#### Anthropic
```python
model_params = {
    'model_name': 'claude-3-sonnet-20240229',
    'anthropic_api_key': 'your-api-key',
    'default_request_timeout': 60,
    'stop_sequences': []
}
llm = get_llm(LlmServers.ANTHROPIC, model_params)
```

#### Groq
```python
model_params = {
    'model_name': 'llama3-8b-8192',
    'groq_api_key': 'your-api-key'
}
llm = get_llm(LlmServers.GROQ, model_params)
```

#### Ollama
```python
model_params = {
    'model_name': 'llama3',
    'llm_base_url': 'http://localhost:11434',
    'format': '',
    'context_window_length': 8192
}
llm = get_llm(LlmServers.OLLAMA, model_params)
```

#### VLLM
```python
model_params = {
    'llm_base_url': 'http://localhost:8000',
    'vllm_api_key': 'your-api-key'
}
llm = get_llm(LlmServers.VLLM, model_params)
```

### Web Search

The `WebSearch` class provides async search capabilities with Tavily:

```python
web_search = WebSearch(
    api_key="your-tavily-api-key",
    search_category="general",  # or use TavilySearchCategory enum
    number_of_days_back=7,
    include_raw_content=True
)

# Search with multiple queries
results = web_search.search(search_queries=[
    "AI developments 2025",
    "machine learning trends"
])
```

### Base Classes

#### ConfigurationBase
Extend this class for configuration management:

```python
from ai_common import ConfigurationBase

class MyConfig(ConfigurationBase):
    def __init__(self):
        super().__init__()
        # Add your configuration logic
```

#### GraphBase
Base class for graph-based AI workflows:

```python
from ai_common import GraphBase

class MyGraph(GraphBase):
    def __init__(self):
        super().__init__()
        # Define your graph structure
```

### Graph Components

The library provides pre-built components for building AI graph workflows:

#### QueryWriter
Generates targeted web search queries using LLM with structured output:

```python
from ai_common.components import QueryWriter

# Initialize with model parameters
model_params = {
    'model': 'gpt-4',
    'model_provider': 'openai',
    'api_key': 'your-api-key',
    'model_args': {}
}

query_writer = QueryWriter(
    model_params=model_params,
    configuration_module_prefix='your_config'
)

# Use in your graph workflow
# Expects state with 'topic' attribute
# Returns state with 'search_queries' attribute
updated_state = query_writer.run(state, config)
```

#### WebSearchNode
Performs web searches and summarizes content using LLM:

```python
from ai_common.components import WebSearchNode

web_search_node = WebSearchNode(
    web_search_api_key='your-tavily-api-key',
    model_params=model_params,
    configuration_module_prefix='your_config'
)

# Use in your graph workflow
# Expects state with 'search_queries' and 'topic' attributes
# Returns state with 'source_str' and 'unique_sources' attributes
updated_state = await web_search_node.run_async(state, config)
# Or use synchronous version:
# updated_state = web_search_node.run(state, config)
```

#### Combined Workflow Example
```python
from ai_common import GraphBase, NodeBase
from ai_common.components import QueryWriter, WebSearchNode

class ResearchGraph(GraphBase):
    def __init__(self, model_params, web_search_api_key, config_prefix):
        super().__init__()
        self.query_writer = QueryWriter(model_params, config_prefix)
        self.web_search_node = WebSearchNode(web_search_api_key, model_params, config_prefix)
    
    def build_graph(self):
        # Define your graph flow
        # 1. Generate queries with QueryWriter
        # 2. Search and summarize with WebSearchNode
        # 3. Continue with additional processing nodes
        pass
```

### Utility Functions

- `tavily_search_async()`: Async web search function
- `deduplicate_and_format_sources()`: Clean and format search results
- `strip_thinking_tokens()`: Remove thinking tokens from LLM responses
- `get_flow_chart()`: Generate flow charts from graph structures
- `load_ollama_model()`: Load and prepare Ollama models

## Requirements

- Python 3.11+
- Dependencies (automatically installed):
  - langchain>=0.3.25
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

## Installation

Add to your `pyproject.toml` or `requirements.txt`:

```toml
[project]
dependencies = [
    "ai-common @ git+https://github.com/bgunyel/ai-common.git@main"
]
```

## Development

For development and testing:

```bash
# Clone the repository
git clone https://github.com/bgunyel/ai-common.git
cd ai-common

# Install development dependencies
uv sync --group test --group lint

# Run tests
uv run pytest

# Run linting
uv run ruff check
```