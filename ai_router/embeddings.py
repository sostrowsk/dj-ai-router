"""
Embeddings client — wraps Azure OpenAI embeddings.

Uses langchain_openai.AzureOpenAIEmbeddings purely as a thin client for
``embed_query``/``embed_documents``; the vector stores themselves (pgvector
via Django ORM, Milvus via pymilvus MilvusClient) are langchain-free.
"""

from django.conf import settings
from langchain_openai import AzureOpenAIEmbeddings


def openai_embeddings() -> AzureOpenAIEmbeddings:
    """Creates an Azure OpenAI embeddings client for vector operations."""
    return AzureOpenAIEmbeddings(
        model="text-embedding-3-large",
        azure_deployment="text-embedding-3-large",
        azure_endpoint=settings.AZURE_EMBEDDINGS_BASE_URL,
        api_key=settings.AZURE_LEASING_API_KEY,
        openai_api_version=settings.AZURE_EMBEDDINGS_API_VERSION,
    )
