from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseEncoder(ABC):
    def __init__(self, name: str = "base_encoder", score_threshold: Optional[float] = None):
        self.name = name
        self.score_threshold = score_threshold

    @abstractmethod
    def __call__(self, texts: List[str]) -> np.ndarray:
        pass


class AzureOpenAIEncoder(BaseEncoder):
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        deployment_name: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2023-05-15",
        score_threshold: Optional[float] = None,
    ):
        super().__init__(name=f"azure_{model}", score_threshold=score_threshold)
        self.model = model
        self.deployment_name = deployment_name or model
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        return self._client

    def __call__(self, texts: List[str]) -> np.ndarray:
        client = self._get_client()
        if isinstance(texts, str):
            texts = [texts]
        response = client.embeddings.create(input=texts, model=self.deployment_name)
        embeddings = [data.embedding for data in response.data]
        return np.array(embeddings)
