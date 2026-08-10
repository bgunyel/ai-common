"""Public API of ai-common, resolved lazily.

Every name in ``__all__`` is imported on first attribute access (PEP 562)
rather than at package-import time. The eager form of this module cost the
same for every caller, because Python runs a parent package's ``__init__``
to completion before it will hand back any submodule of it: ``from
ai_common.enums import ModelNames`` -- two enums and nothing else -- pulled
all nine submodules, and with them langchain, the six provider SDKs, PIL,
tavily and ollama.

Consequences worth knowing before editing this file:

* Cost is per *name*, not per statement. One heavy name in a ``from
  ai_common import ...`` line loads that name's module for the whole
  statement, so a cheap name imported alongside ``get_llm`` buys nothing.
* Resolution is by name, not by module, so a name must be listed in
  ``_MODULE_BY_NAME`` to be reachable at all. ``__all__`` and that mapping
  are pinned to each other by ``test_public_api.py``; adding an export
  means touching both.
* ``import ai_common.llm`` and ``from ai_common.llm import get_llm`` still
  work and still load ``llm``. Laziness applies to the package namespace,
  not to submodule imports.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Never executed. Present so that type checkers, IDEs and `grep` see the
    # same API surface at rest that `__getattr__` produces at runtime.
    from .base import CfgBase, ConfigurationBase, GraphBase, SearchQuery, Queries
    from .engine import Engine
    from .enums import LlmServers, ModelNames, NodeBase, TavilySearchCategory, TavilySearchDepth
    from .llm import load_ollama_model, get_llm, get_model_name_alias
    from .price import calculate_token_cost, calculate_token_cost_for_one_model
    from .utils import (
        get_config_from_runnable,
        get_flow_chart,
        tavily_search_async,
        deduplicate_and_format_sources,
        deduplicate_sources,
        format_sources,
        strip_thinking_tokens,
    )
    from .web_search import WebSearch


#: Exported name -> the submodule that defines it. Grouped by submodule, and
#: ordered cheapest-first, so the cost of reaching for a name is legible here:
#: `enums` is pydantic only, `llm` is every provider SDK.
_MODULE_BY_NAME = {
    'LlmServers': 'enums',
    'ModelNames': 'enums',
    'NodeBase': 'enums',
    'TavilySearchCategory': 'enums',
    'TavilySearchDepth': 'enums',

    'calculate_token_cost': 'price',
    'calculate_token_cost_for_one_model': 'price',

    'CfgBase': 'base',
    'ConfigurationBase': 'base',
    'GraphBase': 'base',
    'SearchQuery': 'base',
    'Queries': 'base',

    'get_config_from_runnable': 'utils',
    'get_flow_chart': 'utils',
    'tavily_search_async': 'utils',
    'deduplicate_and_format_sources': 'utils',
    'deduplicate_sources': 'utils',
    'format_sources': 'utils',
    'strip_thinking_tokens': 'utils',

    'WebSearch': 'web_search',

    'load_ollama_model': 'llm',
    'get_llm': 'llm',
    'get_model_name_alias': 'llm',

    'Engine': 'engine',
}

__all__ = [
    'CfgBase',
    'ConfigurationBase',
    'NodeBase',
    'TavilySearchCategory',
    'TavilySearchDepth',
    'GraphBase',
    'SearchQuery',
    'Queries',
    'LlmServers',
    'ModelNames',
    'Engine',
    'WebSearch',
    'calculate_token_cost',
    'calculate_token_cost_for_one_model',
    'tavily_search_async',
    'load_ollama_model',
    'get_llm',
    'get_model_name_alias',
    'get_flow_chart',
    'deduplicate_and_format_sources',
    'deduplicate_sources',
    'format_sources',
    'strip_thinking_tokens',
    'get_config_from_runnable',
]


def __getattr__(name: str) -> object:
    """Import and return an exported name on first access (PEP 562)."""
    try:
        module_name = _MODULE_BY_NAME[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    from importlib import import_module

    value = getattr(import_module(f'.{module_name}', __name__), name)
    # Bind it in the package namespace: __getattr__ is consulted only on a
    # miss, so each name costs one lookup here and none afterwards.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    # Without this, dir() would report only whatever has been resolved so far,
    # so the visible API surface would depend on import history.
    return sorted(__all__)