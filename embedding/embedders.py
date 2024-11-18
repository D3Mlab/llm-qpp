import torch
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

    def embed(self, text: str):
        # Generate a random tensor of dimension 3
        embedding = torch.tensor([random.random() for _ in range(3)], dtype=torch.float32)
        return embedding

class OpenAIEmbedder(BaseEmbedder):

    def __init__(self, model_name = 'text-embedding-3-large', config={}):
        super().__init__(config = config, model_name = model_name)
        self.api_key = os.environ.get('OPENAI_API_KEY', config.get('api_key'))


    def embed(self, text: str):
        try:
            # Use OpenAI API to get embeddings
            response = openai.Embedding.create(input=text, model=self.model)
            embedding = response['data'][0]['embedding']
            return torch.tensor(embedding, dtype=torch.float32)
        except openai.error.OpenAIError as e:
            self.logger.error(f"An error occurred while fetching embeddings: {e}")
            return None



class HuggingFaceEmbedder(BaseEmbedder):

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', config={}, pooling_type='mean', use_gpu=True):
        super().__init__(config=config, model_name=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling_type = pooling_type
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def embed(self, text: str):
        try:
            # Tokenize the input text
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True).to(self.device)
            if inputs['input_ids'].shape[-1] == self.tokenizer.model_max_length:
                self.logger.warning("Input text was truncated to fit the model's maximum length.")
            with torch.no_grad():
                # Get the model's last hidden states
                outputs = self.model(**inputs)
                if self.pooling_type == 'mean':
                    # Mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                elif self.pooling_type == 'cls':
                    # CLS token pooling
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                else:
                    raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
            return embedding.cpu()
        except Exception as e:
            self.logger.error(f"An error occurred while fetching embeddings: {e}")
            return None

