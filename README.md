# ai-common

A Python utility library providing common classes and functions for building
LLM-powered AI applications and agentic graph workflows.

`ai-common` gives you a single, provider-agnostic surface over the major LLM
backends (Anthropic, OpenAI, Google, Groq, Ollama, OpenRouter), Tavily web
search, reusable LangGraph-style workflow components, token-cost accounting,
and a set of small utilities for research/RAG pipelines.

## Features

- **Multi-LLM support** — one `get_llm()` entry point for Anthropic, OpenAI,
  Google (Gemini), Groq, Ollama (Cloud), and OpenRouter, returning a LangChain
  `BaseChatModel`.
- **Model-name aliasing** — refer to a model by a single `ModelNames` enum and
  let the library resolve the provider-specific string (e.g. `gpt-oss-120b` →
  `openai/gpt-oss-120b` on Groq, `gpt-oss:120b` on Ollama).
- **Web search** — async Tavily search with automatic source deduplication and
  formatting.
- **Graph components** — pre-built `QueryWriter` and `WebSearchNode` for
  research/RAG workflows.
- **Base abstractions** — configuration and graph base classes plus structured
  `SearchQuery`/`Queries` models.
- **Token-cost accounting** — compute USD cost from token usage using a
  built-in price table.
- **Engine framework** — a thin orchestration layer that runs a responder
  graph, tracks history, and persists responses.
- **Utilities** — flow-chart generation, source formatting, and thinking-token
  stripping.

## Requirements

- **Python 3.13+**
- Runtime dependencies (installed automatically):
  - `langchain>=1.0.0`, `langchain-core>=1.3.3`
  - `langchain-anthropic>=1.0.0`, `langchain-google-genai>=4.2.1`,
    `langchain-groq>=1.0.0`, `langchain-ollama>=1.0.0`,
    `langchain-openai>=1.0.1`, `langchain-openrouter>=0.2.6`
  - `ollama>=0.6.0`, `openai>=2.6.0`
  - `pillow>=11.3.0`, `tavily-python>=0.7.12`, `tqdm>=4.67.1`

## Installation

Add `ai-common` as a Git dependency in your `pyproject.toml`:

```toml
[project]
dependencies = [
    "ai-common @ git+https://github.com/bgunyel/ai-common.git@main",
]
```

Or with `uv`:

```bash
uv add "ai-common @ git+https://github.com/bgunyel/ai-common.git@main"
```

## Quick Start

```python
from pydantic import SecretStr
from ai_common import get_llm, LlmServers, ModelNames, WebSearch

# 1. Get a provider-agnostic LLM (returns a LangChain BaseChatModel)
llm = get_llm(
    model_name=ModelNames.GPT_5_1,
    model_provider=LlmServers.OPENAI,
    api_key=SecretStr("your-openai-api-key"),
    model_args={"reasoning_effort": "high"},
)
response = llm.invoke("Summarize the latest advances in retrieval-augmented generation.")

# 2. Run an async Tavily web search
import asyncio
from ai_common import TavilySearchCategory, TavilySearchDepth

web_search = WebSearch(api_key=SecretStr("your-tavily-api-key"))
unique_sources = asyncio.run(web_search.search(
    search_queries=["AI trends 2026", "agentic workflows"],
    search_category=TavilySearchCategory.GENERAL,
    search_depth=TavilySearchDepth.ADVANCED,
    chunks_per_source=3,
    number_of_days_back=30,
    max_results_per_query=5,
    include_images=False,
    include_image_descriptions=False,
    include_favicon=False,
))
# -> dict keyed by URL: { "https://...": {"title": ..., "content": ..., ...}, ... }
```

## Public API

Everything below is importable directly from the top-level `ai_common` package,
except the graph components, which live in `ai_common.components`.

These names resolve lazily (PEP 562): `import ai_common` costs nothing, and each
name pulls only the submodule that defines it, on first access. Cost is therefore
per *name*, not per package — `from ai_common import ModelNames` stays cheap,
while any statement mentioning `get_llm` loads the provider SDKs.

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `get_llm` | function | Build a LangChain chat model for any supported provider |
| `get_model_name_alias` | function | Resolve a `ModelNames` enum to a provider-specific model string |
| `load_ollama_model` | function | Pull and warm an Ollama model into memory |
| `LlmServers` | enum | Supported providers |
| `ModelNames` | enum | Known model identifiers |
| `WebSearch` | class | Async Tavily search with deduplication |
| `Engine` | class | Orchestrates a responder graph, history, and persistence |
| `GraphBase`, `CfgBase`, `ConfigurationBase` | class | Base classes for graphs/config |
| `SearchQuery`, `Queries` | model | Structured search-query models |
| `NodeBase`, `TavilySearchCategory`, `TavilySearchDepth` | enum | Workflow/search constants |
| `calculate_token_cost`, `calculate_token_cost_for_one_model` | function | USD cost from token usage |
| `tavily_search_async` | function | Low-level async Tavily search |
| `deduplicate_sources`, `format_sources`, `deduplicate_and_format_sources` | function | Source post-processing |
| `strip_thinking_tokens` | function | Remove `<think>...</think>` spans |
| `get_flow_chart` | function | Render a graph to a PNG image |
| `get_config_from_runnable` | function | Load a `Configuration` from a `RunnableConfig` |
| `QueryWriter`, `WebSearchNode` | class | Pre-built graph components (`ai_common.components`) |

## LLM Configuration

`get_llm()` has a single, uniform signature across all providers:

```python
def get_llm(
    model_name: ModelNames,
    model_provider: LlmServers,
    api_key: SecretStr,
    model_args: dict[str, Any],
) -> BaseChatModel
```

- `model_name` — a `ModelNames` enum value (not a raw string).
- `model_provider` — an `LlmServers` enum value.
- `api_key` — a `pydantic.SecretStr`.
- `model_args` — provider-specific keyword arguments (temperature, top_p,
  reasoning effort, …). The function normalizes common keys per provider (see
  notes below).

Supported `LlmServers`: `ANTHROPIC`, `GOOGLE`, `GROQ`, `OLLAMA`, `OPENAI`,
`OPENROUTER`. `VLLM` is reserved but currently raises `NotImplementedError`.

### OpenAI

```python
llm = get_llm(
    model_name=ModelNames.GPT_5_1,
    model_provider=LlmServers.OPENAI,
    api_key=SecretStr("your-api-key"),
    model_args={"reasoning_effort": "high"},
)
```

Uses the Responses API (`use_responses_api=True`). `reasoning_effort` is mapped
to a `reasoning={"effort": ..., "summary": "auto"}` block; when effort is not
`"none"`, `temperature`/`top_p`/`logprobs` are dropped (unsupported with
reasoning).

### Anthropic

```python
llm = get_llm(
    model_name=ModelNames.GPT_5_1,          # use any ModelNames value your app defines
    model_provider=LlmServers.ANTHROPIC,
    api_key=SecretStr("your-api-key"),
    model_args={"temperature": 0, "max_tokens": 4096},
)
```

`model_args` are forwarded to `ChatAnthropic` (with `stop=None`,
`timeout=None`).

### Google (Gemini)

```python
llm = get_llm(
    model_name=ModelNames.GEMINI_3_5_FLASH,
    model_provider=LlmServers.GOOGLE,
    api_key=SecretStr("your-api-key"),
    model_args={"reasoning_effort": "high"},
)
```

For `gemini-3*` models, `temperature` is forced to `1.0`. `reasoning` /
`reasoning_effort` are mapped to `thinking_level`.

### Groq

```python
llm = get_llm(
    model_name=ModelNames.GPT_OSS_120B,
    model_provider=LlmServers.GROQ,
    api_key=SecretStr("your-api-key"),
    model_args={"top_p": 0.95, "reasoning_effort": "high"},
)
```

Runs with `service_tier="auto"`. `top_p` is moved into `model_kwargs`;
`reasoning` is mapped to `reasoning_effort`.

### Ollama (Cloud)

```python
llm = get_llm(
    model_name=ModelNames.GLM_5,
    model_provider=LlmServers.OLLAMA,
    api_key=SecretStr("your-ollama-api-key"),
    model_args={"temperature": 0, "reasoning_effort": "high"},
)
```

Targets Ollama Cloud (`https://ollama.com`) with a Bearer-token header derived
from `api_key`. `reasoning_effort` is mapped to `reasoning`.

### OpenRouter

```python
llm = get_llm(
    model_name=ModelNames.DEEPSEEK_V_4_FLASH,
    model_provider=LlmServers.OPENROUTER,
    api_key=SecretStr("your-openrouter-api-key"),
    model_args={"temperature": 0, "top_p": 0.95, "reasoning_effort": "high"},
)
```

`temperature` (default `0`), `top_p` (default `0.95`), and `reasoning_effort`
are pulled out explicitly; any remaining `model_args` are passed via
`model_kwargs`.

### Model-name aliasing

The same logical model is served under different names on different platforms.
Refer to it once via `ModelNames` and let the library resolve the alias:

```python
from ai_common import get_model_name_alias, LlmServers, ModelNames

get_model_name_alias(ModelNames.GPT_OSS_120B, LlmServers.GROQ)    # -> "openai/gpt-oss-120b"
get_model_name_alias(ModelNames.GPT_OSS_120B, LlmServers.OLLAMA)  # -> "gpt-oss:120b"
```

`get_llm()` calls this internally, so you never pass provider-specific strings.

## Web Search

`WebSearch` wraps Tavily's async client and returns deduplicated sources keyed
by URL.

```python
from pydantic import SecretStr
from ai_common import WebSearch, TavilySearchCategory, TavilySearchDepth

web_search = WebSearch(api_key=SecretStr("your-tavily-api-key"))

unique_sources = await web_search.search(
    search_queries=["AI developments 2026", "machine learning trends"],
    search_category=TavilySearchCategory.GENERAL,   # GENERAL | NEWS | FINANCE
    search_depth=TavilySearchDepth.ADVANCED,        # BASIC | ADVANCED
    chunks_per_source=3,
    number_of_days_back=7,
    max_results_per_query=5,
    include_images=False,
    include_image_descriptions=False,
    include_favicon=False,
)
```

For lower-level control there is also `tavily_search_async(...)`, plus the
post-processing helpers `deduplicate_sources()`, `format_sources()`, and
`deduplicate_and_format_sources()`.

## Token-Cost Accounting

Compute the USD cost of a run from provider/model metadata and token usage. The
price table lives in `ai_common.price.PRICE_USD_PER_MILLION_TOKENS`.

```python
from ai_common import calculate_token_cost, LlmServers, ModelNames

llm_config = {
    "orchestrator_model": {"model_provider": LlmServers.OPENAI, "model": ModelNames.GPT_5_1},
    "writer_model":       {"model_provider": LlmServers.GROQ,   "model": ModelNames.GPT_OSS_120B},
}
token_usage = {
    ModelNames.GPT_5_1:      {"input_tokens": 12_000, "output_tokens": 3_000},
    ModelNames.GPT_OSS_120B: {"input_tokens":  8_000, "output_tokens": 2_000},
}

cost_list, total_cost = calculate_token_cost(llm_config, token_usage)
# cost_list -> [{"model_provider": ..., "model": ..., "cost": ...}, ...]
# total_cost -> float (USD)
```

Use `calculate_token_cost_for_one_model(params, token_usage)` for a single
model.

## Base Classes & Structured Models

```python
from ai_common import GraphBase, ConfigurationBase, CfgBase, SearchQuery, Queries
```

- **`GraphBase`** — abstract base for graph workflows; implement `__init__`,
  `build_graph`, and `get_response`.
- **`ConfigurationBase`** — dataclass config base with
  `from_runnable_config()`, hydrating fields from env vars / a `RunnableConfig`.
- **`CfgBase`** — Pydantic config base with `from_runnable()`.
- **`SearchQuery`** — a structured query (`search_query`, `aspect`,
  `rationale`); **`Queries`** wraps a list of them, suitable as an LLM
  structured-output schema.

## Graph Components

Pre-built nodes for research/RAG graphs live in `ai_common.components`. Both
take a `model_params` dict of the shape:

```python
model_params = {
    "model": ModelNames.GPT_5_1,          # ModelNames enum
    "model_provider": LlmServers.OPENAI,  # LlmServers enum
    "api_key": SecretStr("your-api-key"),
    "model_args": {"reasoning_effort": "high"},
}
```

### QueryWriter

Generates targeted web-search queries for a topic.

```python
from ai_common.components import QueryWriter

query_writer = QueryWriter(
    model_params=model_params,
    configuration_module_prefix="your_app.configuration",
)

# Async helper — returns {"search_queries": [SearchQuery, ...], "token_usage": {...}}
result = await query_writer.generate_queries(topic="agentic RAG", number_of_queries=5)
```

> Note: the graph-style `run(state, config)` method is a work in progress and
> currently raises `NotImplementedError`; use `generate_queries(...)` for now.

### WebSearchNode

Searches the web for the state's queries and summarizes each source with the
configured LLM.

```python
from ai_common.components import WebSearchNode

web_search_node = WebSearchNode(
    web_search_api_key=SecretStr("your-tavily-api-key"),
    model_params=model_params,
    configuration_module_prefix="your_app.configuration",
)

# Expects state with 'search_queries' and 'topic'; returns state with
# 'source_str' and 'unique_sources' populated.
state = await web_search_node.run_async(state, config)
# Synchronous wrapper:
# state = web_search_node.run(state, config)
```

Both components read runtime search settings (category, depth, days back,
max results per query, etc.) from a `Configuration` object resolved via
`get_config_from_runnable()` using your `configuration_module_prefix`.

## Engine

`Engine` is a thin orchestration layer around a `GraphBase` responder that
tracks conversation history and persists each response to disk.

```python
from ai_common import Engine, LlmServers

engine = Engine(
    responder=my_graph,               # a GraphBase implementation
    llm_server=LlmServers.OLLAMA,     # if OLLAMA, listed models are pre-loaded
    models=["glm-5:cloud"],
    llm_base_url="https://ollama.com",
    save_to_folder="/path/to/responses",
)

response = engine.get_response(input_dict={"question": "..."})
engine.save_flow_chart(save_to_folder="/path/to/diagrams")   # writes flow_chart.png
```

## Utility Functions

- `tavily_search_async(...)` — low-level concurrent Tavily search.
- `deduplicate_sources(...)` / `format_sources(...)` /
  `deduplicate_and_format_sources(...)` — clean and format search results.
- `strip_thinking_tokens(text)` — remove `<think>...</think>` spans from LLM
  output.
- `get_flow_chart(rag_model)` — render a graph to a `PIL.Image` (Mermaid PNG).
- `get_config_from_runnable(prefix, config)` — load a `Configuration` from a
  `RunnableConfig`.
- `load_ollama_model(model_name, ollama_url)` — pull and warm an Ollama model.

## Project Layout

```
ai-common/
├── src/ai_common/          # the installable library (published package)
│   ├── base.py             # CfgBase, ConfigurationBase, GraphBase, SearchQuery, Queries
│   ├── enums.py            # LlmServers, ModelNames, Tavily* , NodeBase
│   ├── llm.py              # get_llm, get_model_name_alias, load_ollama_model
│   ├── price.py            # token-cost accounting + price table
│   ├── web_search.py       # WebSearch
│   ├── utils.py            # search/formatting/flow-chart utilities
│   ├── engine.py           # Engine
│   └── components/         # QueryWriter, WebSearchNode
├── src/tests/              # public-API tests
└── scripts/                # local dev/tooling scripts (isolated dependencies)
```

### `scripts/` dependency isolation

The `scripts/` folder holds local development and tooling code that is
**deliberately kept separate** from the published library. Its dependencies
live in a dedicated `scripts` dependency group in `pyproject.toml`, so they are
**not** part of the library's runtime dependencies and are never pulled in by
consumers who install `ai-common`:

```toml
[dependency-groups]
scripts = ["pydantic-settings>=2.14.2", "rich>=15.0.0"]
```

Install and run scripts with the group enabled:

```bash
uv sync --group scripts
uv run --group scripts python scripts/main_dev.py
```

## Development

```bash
# Clone
git clone https://github.com/bgunyel/ai-common.git
cd ai-common

# Install all dependency groups (main + test + lint + scripts)
uv sync --all-groups

# Run tests
make test            # or: uv run --group test pytest src/tests/

# Lint
uv run ruff check
```

### Dependency-security targets

The `Makefile` provides a layered supply-chain workflow. All scanning/sync
targets operate across every dependency group (`--all-groups`) for complete
coverage:

| Target | What it does |
| --- | --- |
| `make audit` | Tier 1 — scan the committed `uv.lock` against OSV/GHSA advisories (fast, read-only). Requires [`osv-scanner`](https://github.com/google/osv-scanner). |
| `make scan` | Tier 2 — [GuardDog](https://github.com/DataDog/guarddog) static analysis on every locked dep, via the `guarddog-cached` console script shipped by this package. |
| `make verify` | Combined tier-1 + tier-2 sweep against the committed lock (release gate). |
| `make upgrade-safe` | Resolve a candidate upgrade, run **both** scanners on it, and revert `uv.lock` if either fires; otherwise sync. |
| `make upgrade` | Blind upgrade with only a 7-day quarantine (`--exclude-newer`); prefer `upgrade-safe`. |

#### The shared GuardDog cache

`guarddog-cached` is a console script shipped by this package, so a project
gets the tier-2 wrapper by depending on ai-common rather than by copying a
script into its own `scripts/`.

A scan result is a fact about PyPI — package X at version Y, judged by
GuardDog Z — not a fact about any one project, so results are cached in
`$XDG_CACHE_HOME/guarddog-cached/cache.json` (default `~/.cache/…`) and
**every project on the machine reuses every other project's scans**.

Because the cache is shared, it is written defensively: entries are keyed on
all three of (name, version, guarddog_version), so upgrading GuardDog
re-scans under the new version instead of invalidating results other
projects still rely on; a save re-reads and merges under an exclusive lock,
so concurrent projects cannot drop each other's results; and the file is
replaced by rename, so a killed run never leaves a partial cache behind.

Beside the cache, `reports/` keeps GuardDog's full report for each entry,
in a file named after the same key (`six==1.17.0@3.1.0.json`). A cache entry
is a summary built for the gate — it records that a rule fired and where,
but not the text it fired on, which is the one thing needed to review a
finding before waiving it. `make scan` names the report for any package it
blocks. The reports are evidence for a human and never an input to the
verdict, so the directory can be deleted at any time.

Ctrl-C is safe and useful: every completed scan is persisted immediately, so
a long sweep can be done in short sittings and picks up where it stopped.
`uv.lock` is never left half-upgraded — the candidate lock is written whole
before scanning begins, and an interrupted `upgrade-safe` restores the
original.

#### Scanning on a time budget

A first sweep on a new GuardDog version re-scans everything and can take an
hour. To do it in slices:

```sh
make upgrade-safe GUARDDOG_BUDGET=600     # 10 minutes, then stop
```

Scanning stops starting new packages once the budget is spent and exits 75,
which reverts `uv.lock` exactly as an interrupt would. Completed scans stay
cached, so repeated budgeted runs converge on a full sweep and only a run
that finishes inside its budget can adopt an upgrade. The budget bounds when
a scan *starts*, not when it ends, so a slow package may overshoot by one.

#### Why the gate does not use GuardDog's exit code

`guarddog pypi scan` exits 0 whether it found nothing, found three malicious
indicators, or never managed to download the package. Nonzero means only that
GuardDog was *called* wrong. Gating on it would let a scan that never ran
count as a pass.

So the wrapper scans with `--output-format=json` and derives its own verdict:

| verdict | condition | gate |
| --- | --- | --- |
| `INCOMPLETE` | `errors` non-empty — some rules did not run | **fails** |
| `BLOCKED` | a risk at severity `high` | **fails** |
| `advisory` | only `low`/`medium` risks | passes, reported |
| `clean` | no risks | passes |

#### Why severity, and not a list of rule names

The gate used to block on seven named rules. GuardDog 3 renamed all 61 of its
rules onto a `capability-*`/`threat-*` taxonomy, **none of the seven
survived**, and the gate quietly stopped blocking anything — it asked "is this
name in my list?", got "no" for every rule GuardDog now emits, and passed
everything. Nothing announced it.

So the verdict now rests on `risks[].severity`, a three-value vocabulary
(`low`, `medium`, `high`) that GuardDog derives itself. A risk's severity is
its threat rule's severity, downgraded one level when the correlating
capability sits in another file and two when it sits in another category — so
`high` means a high-severity rule that either stands alone (install-time, or
specific enough to be malware-only) or correlates inside a single file.

**Anything the wrapper does not understand blocks rather than passes.** An
unrecognised severity is treated as blocking, and a scan that completed but
whose report has no `risks` field is `INCOMPLETE`. A gate that stops
understanding its input has to fail noisily; the previous one failed silently,
which is the only outcome that matters here.

GuardDog's own headline `risk_score.label` is reported and deliberately not
acted on. Measured 2026-08-11: tqdm scores **7.2/10 `high_risk`** for naming
`api.telegram.org` in a file called `contrib/telegram.py`, and pyyaml **8.8**.
Gating on the label would block two of six ordinary packages.

**Only complete scans are cached.** A scan that reported `errors` is retried
next run rather than frozen, so a transient network failure heals itself
instead of becoming a permanent machine-wide clean bill.

The verdict is computed when an entry is *read*, so changing
`BLOCKING_SEVERITY` or accepting a finding re-decides every cached package
without re-scanning.

#### Accepting a reviewed finding

`accepted.json`, beside the cache, waives named rules for one package version
across every project on the machine. A waiver names either the threat rule or
the risk it rolls up into — the rule is narrower and usually what you want:

```json
{
  "schema": 1,
  "accepted": {
    "somepkg==1.2.3": {
      "rules": ["threat-runtime-obfuscation-steganography"],
      "reason": "matches a base64 fixture in the package's own test suite, reviewed 2026-08-11",
      "by": "bgunyel",
      "at": "2026-08-11"
    }
  }
}
```

Waivers are keyed on (name, version) without the GuardDog version — the
review was of the package's code, which a GuardDog upgrade does not change.
A *new* package version is never covered by an old waiver. Because the file
is machine-wide rather than per-repository, a waiver does not pass through
code review; the `reason`/`by`/`at` fields are there so the decision is at
least auditable after the fact.