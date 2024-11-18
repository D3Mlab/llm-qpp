import torch
import random
import os
import openai
from abc import ABC, abstractmethod
from utils.setup_logging import setup_logging

class BaseEmbedder(ABC):

    def __init__(self, config = {}):

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

    @abstractmethod
    def embed(self, text: str):
        #return torch embedding tensor for text
         raise NotImplementedError("This method must be implemented by a subclass.")

class RandomEmbedder(BaseEmbedder):

    def __init__(self, config={}):
        super().__init__(config)

    def embed(self, text: str):
        # Generate a random tensor of dimension 3
        embedding = torch.tensor([random.random() for _ in range(3)], dtype=torch.float32)
        return embedding

class OpenAIEmbedder(BaseEmbedder):

    def __init__(self, config={}):
        super().__init__(config)
        self.model = config.get('model', 'text-embedding-3-large')  # Default OpenAI model to use
        self.api_key = os.environ.get('OPENAI_API_KEY', config.get('api_key'))


    def embed(self, text: str):
        # Use OpenAI API to get embeddings
        response = openai.Embedding.create(input=text, model=self.model)
        embedding = response['data'][0]['embedding']
        return torch.tensor(embedding, dtype=torch.float32)