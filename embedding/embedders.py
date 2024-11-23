import torch
import torch.nn.functional as F
import random
import os
import openai
from utils.setup_logging import setup_logging
from transformers import AutoTokenizer, AutoModel
from abc import ABC, abstractmethod
from utils.setup_logging import setup_logging

class BaseEmbedder(ABC):

    def __init__(self, config = {}, model_name = ''):

        self.model = model_name
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

    @abstractmethod
    def embed(self, text: str):
        #return torch embedding tensor for text
         raise NotImplementedError("This method must be implemented by a subclass.")

class RandomEmbedder(BaseEmbedder):
    #sanity check class
    def __init__(self, config={}, model_name = ''):
        super().__init__(config)

    def embed(self, text):
        if isinstance(text, str):
            # Generate a random tensor of dimension 3 for a single string
            embedding = torch.tensor([random.random() for _ in range(3)], dtype=torch.float32)
            return [embedding]  # Return as a list for consistent handling
        elif isinstance(text, list):
            # Generate random tensors for a batch of strings
            embedding = [torch.tensor([random.random() for _ in range(3)], dtype=torch.float32) for _ in text]
            return embedding
        else:
            raise ValueError("Input must be either a string or a list of strings.")


class TestQueryEmbedder(BaseEmbedder):
    #returns a predefined embedding for a query for testing
    def __init__(self, config={}, model_name = ''):
        super().__init__(config)

    def embed(self, text):
        
            # Generate a random tensor of dimension 3 for a single string
            embedding = F.normalize(torch.tensor([2, 3, 4], dtype=torch.float32),p=2,dim=0)
            return [embedding]  # Return as a list for consistent handling


class OpenAIEmbedder(BaseEmbedder):
    #this api doesn't support batching! use OpenAI batch API for batching. 
    def __init__(self, model_name = 'text-embedding-3-large', config={}, use_gpu=True):
        super().__init__(config = config, model_name = model_name)
        self.api_key = os.environ.get('OPENAI_API_KEY', config.get('api_key'))
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

    def embed(self, text: str):
        if not isinstance(text, str):
            #ensure we're not getting a batch of size > 1
            if len(text) == 1:
                text = text[0]
            else:
                raise ValueError("Input must be a single string or singleton list with one string.")
        try:
            # Use OpenAI API to get embeddings
            response = openai.Embedding.create(input=text, model=self.model)
            embedding = response['data'][0]['embedding']
            return [torch.tensor(embedding, dtype=torch.float32, device = self.device)]
        except openai.error.OpenAIError as e:
            self.logger.error(f"An error occurred while fetching embeddings: {e}")
            return None


class HuggingFaceEmbedder(BaseEmbedder):

    def __init__(self, *, model_name='sentence-transformers/all-MiniLM-L6-v2', config={}, pooling_type='mean', use_gpu=True):
        super().__init__(config=config, model_name=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.pooling_type = pooling_type

        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if use_gpu and torch.cuda.is_available():
            self.logger.info("Using GPU")
        else:
            self.logger.info("Using CPU")

    def embed(self, texts):
        if isinstance(texts, str):
            self.logger.debug('converting string to singleton list')    
            texts = [texts]
        try:
            # Tokenize the input text batch
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
            if inputs['input_ids'].shape[-1] == self.tokenizer.model_max_length:
                self.logger.warning("Some input texts were truncated to fit the model's maximum length.")
            with torch.no_grad():
                # Get the model's last hidden states
                outputs = self.model(**inputs)
                if self.pooling_type == 'mean':
                    # Mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1)
                elif self.pooling_type == 'cls':
                    # CLS token pooling
                    embedding = outputs.last_hidden_state[:, 0, :]
                else:
                    raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding
        except Exception as e:
            self.logger.error(f"An error occurred while fetching embeddings: {e}")
            return None

     