"""
Unified LLM client factory — replaces LangChain get_chat_client().

Usage:
    from ai_router.client import get_llm_client
    client = get_llm_client()  # uses default model
    result, parsed = client.invoke(system_prompt, user_prompt, output_schema=MySchema)

    # Streaming:
    for chunk in client.stream(system_prompt, user_prompt):
        print(chunk, end="")

    # PDF extraction with caching:
    result, parsed = client.invoke_with_pdf_cache(pdf_path, system_prompt, user_prompt, output_schema=MySchema)
"""

from ai_router.cached_llm import CachedAnthropicClient, CachedGeminiClient, get_cached_client


def get_llm_client(model: str = None) -> CachedAnthropicClient | CachedGeminiClient:
    """
    Get an LLM client for the given model.

    Drop-in replacement for the old LangChain-based get_chat_client().
    Returns a CachedAnthropicClient or CachedGeminiClient with:
    - invoke(system_prompt, user_prompt, output_schema=None)
    - stream(system_prompt, user_prompt)
    - invoke_with_cache(document_content, document_name, extraction_prompt, ...)
    - invoke_with_pdf_cache(pdf_path, system_prompt, extraction_prompt, ...)
    """
    return get_cached_client(model)
