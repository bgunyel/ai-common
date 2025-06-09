from .enums import LlmServers

PRICE_USD_PER_MILLION_TOKENS = {
    LlmServers.GROQ.value: {
        'deepseek-r1-distill-llama-70b': {'input_tokens': 0.75, 'output_tokens': 0.99},
        'gemma2-9b-it': {'input_tokens': 0.20, 'output_tokens': 0.20},
        'llama3-70b-8192': {'input_tokens': 0.59, 'output_tokens': 0.79},
        'llama3-8b-8192': {'input_tokens': 0.05, 'output_tokens': 0.08},
        'llama-3.1-8b-instant': {'input_tokens': 0.05, 'output_tokens': 0.08},
        'llama-3.3-70b-versatile': {'input_tokens': 0.59, 'output_tokens': 0.79},
        'meta-llama/llama-4-maverick-17b-128e-instruct': {'input_tokens': 0.20, 'output_tokens': 0.60},
        'meta-llama/llama-4-scout-17b-16e-instruct': {'input_tokens': 0.11, 'output_tokens': 0.34},
        'meta-llama/llama-guard-4-12b': {'input_tokens': 0.20, 'output_tokens': 0.20},
        'mistral-saba-24b': {'input_tokens': 0.79, 'output_tokens': 0.79},
        'qwen-qwq-32b': {'input_tokens': 0.29, 'output_tokens': 0.39}
    },
    LlmServers.OPENAI.value: {
        'gpt-4.1': {'input_tokens': 2.00, 'output_tokens': 8.00},
        'gpt-4.1-mini': {'input_tokens': 0.40, 'output_tokens': 1.60},
        'gpt-4.1-nano': {'input_tokens': 0.10, 'output_tokens': 0.40},
        'gpt-4o': {'input_tokens': 2.50, 'output_tokens': 10.00},
        'gpt-4o-mini': {'input_tokens': 0.15, 'output_tokens': 0.60},
        'o1': {'input_tokens': 15.00, 'output_tokens': 60.00},
        'o3': {'input_tokens': 10.00, 'output_tokens': 40.00},
        'o1-mini': {'input_tokens': 1.10, 'output_tokens': 4.40},
        'o3-mini': {'input_tokens': 1.10, 'output_tokens': 4.40},
    },
    LlmServers.ANTHROPIC.value: {
        'claude-opus-4-latest': {'input_tokens': 15.00, 'output_tokens': 75.00},
        'claude-sonnet-4-latest': {'input_tokens': 3.00, 'output_tokens': 15.00},
        'claude-3-5-haiku-latest': {'input_tokens': 0.80, 'output_tokens': 4.00},
        'claude-3-7-sonnet-latest': {'input_tokens': 3.00, 'output_tokens': 15.00},
    }
}