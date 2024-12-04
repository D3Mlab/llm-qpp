from abc import ABC, abstractmethod
import os
import time
import importlib
import openai
from dotenv import load_dotenv
from utils.setup_logging import setup_logging
import yaml
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import GoogleAPICallError, RetryError, InvalidArgument


class BaseLLM(ABC):

    def __init__(self,config, model_name = ""):
        #model_name: e.g. gpt-3.5-turbo-1106

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

        self.model_name = model_name

        llm_config = self.config.get('llm', {})
        self.dwell_time = llm_config.get('dwell_time', 60) 
        self.num_retries = llm_config.get('num_retries', 5)
        
    def prompt(self, *args, **kwargs):
        """Prompts with retries and catches errors"""
        attempt = 0
        while attempt < self.num_retries:
            try:
                self.logger.debug("calling llm api")
                return self.call_api(*args, **kwargs)
            except Exception as e:
                self.handle_exception(e, attempt)
                attempt += 1
                if attempt < self.num_retries:
                    self.logger.info(f"Retrying in {self.dwell_time} seconds...")
                    time.sleep(self.dwell_time)
                else:
                    self.logger.error("All retry attempts exhausted.")
                    return {"error": str(e)}

    def handle_exception(self, e, attempt):
        """Default exception handler, can be overridden by subclasses."""
        self.logger.info(f"Attempt {attempt + 1} failed: {e}")
        pass

    @abstractmethod
    def call_api(self, prompt):
        #attempt to call llm and return dict of response results
        #e.g. {message: "Hi", logprobs: (2.4,0.4)}
        """Method to be implemented by subclasses to make the actual API call."""
        raise NotImplementedError("This method must be implemented by a subclass.")



class OpenAILLM(BaseLLM):
    def __init__(self, config, model_name):
        super().__init__(config,model_name)
        openai.api_key = os.environ['OPENAI_API_KEY']

    def call_api(self, prompt, temp = 0.0, logprobs = False, top_logprobs = 3):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            logprobs=logprobs,
        )
        return {
            "message": response.choices[0].message['content'],
            "logprobs": response.choices[0].logprobs if logprobs == True else None
        }



class GeminiLLM(BaseLLM):
    def __init__(self, config, model_name):
        super().__init__(config,model_name)
        self.GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
        genai.configure(api_key = self.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.model_name)
        self.safety_settings = {HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,}

    def call_api(self, prompt, temp = 0.0):
        generation_config=genai.types.GenerationConfig(
            temperature=temp)
        response = self.model.generate_content(prompt, safety_settings = self.safety_settings, generation_config=generation_config)
        return {
            "message": response.text    
        }

    def handle_exception(self, e, attempt):
        ##todo -- see gemini_wrapper.py code for key shuffling example given certain error types
        self.logger.info(f"Attempt {attempt + 1} failed: {e}")
        pass
