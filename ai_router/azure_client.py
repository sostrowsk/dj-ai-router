from django.conf import settings

from ai_router.encoders import AzureOpenAIEncoder

AZURE_MODEL_CONFIG = {
    "gpt-5.4": {
        "provider": "azure",
        "engine": "openai",
        "api_key": "AZURE_SHOOBRIDGE_API_KEY",
        "api_version": "AZURE_CHAT_API_VERSION",
        "endpoint": "AZURE_SHOOBRIDGE_BASE_URL",
        "supports_temp": True,
    },
}


def openai_encoder() -> AzureOpenAIEncoder:
    """Creates a client for encoding text with Azure OpenAI."""
    return AzureOpenAIEncoder(
        model="text-embedding-3-large",
        deployment_name="text-embedding-3-large",
        azure_endpoint=settings.AZURE_EMBEDDINGS_BASE_URL,
        api_key=settings.AZURE_LEASING_API_KEY,
        api_version=settings.AZURE_EMBEDDINGS_API_VERSION,
    )


def openai_embeddings():
    """Creates Azure OpenAI embeddings client for Milvus vector store."""
    from ai_router.embeddings import openai_embeddings as _impl

    return _impl()
