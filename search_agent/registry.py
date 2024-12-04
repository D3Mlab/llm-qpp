# registry.py
from .general_agent import GeneralAgent
from .dense_retriever import DenseRetriever
from llm.prompter import Prompter
from .policies import PipelinePolicy
from utils.utils import AgentLogic

# Classes that function as agents (have a rank method)
AGENT_CLASSES = {
    'DenseRetriever': DenseRetriever,
    'GeneralAgent': GeneralAgent,
}

# Classes that might be used as components in a general agent (QPP, Embedders, LLMs, etc.)
COMPONENT_CLASSES = {
    'AgentLogic': AgentLogic,
    'Prompter': Prompter
    }
COMPONENT_CLASSES.update(AGENT_CLASSES)

POLICY_CLASSES = {
    'PipelinePolicy': PipelinePolicy,
}