from .general_agent import GeneralAgent
from .dense_retriever import DenseRetriever
from .policies import PipelinePolicy
import embedding


#Classes that function as agents (have a rank method)
AGENT_CLASSES = {
    'DenseRetriever': DenseRetriever,
    'GeneralAgent': GeneralAgent
    }

#Classes that might be used as components in a general agent (QPP, Embedders, LLMs, etc)
COMPONENT_CLASSES = {}
COMPONENT_CLASSES.update(AGENT_CLASSES)

#"Main" methods for a class. E.g. rank() for a BaseAgent, query_performance_prediction() for QPP, etc  
MAIN_ACTIONS = {
    DenseRetriever: DenseRetriever.rank 
    }

POLICY_CLASSES = {
    'PipelinePolicy' : PipelinePolicy
    }