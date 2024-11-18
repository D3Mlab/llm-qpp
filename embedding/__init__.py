from .embedders import RandomEmbedder, OpenAIEmbedder, HuggingFaceEmbedder

EMBEDDER_CLASSES = {
    'HuggingFaceEmbedder': HuggingFaceEmbedder,
    'RandomEmbedder':RandomEmbedder,
    'HuggingFaceEmbedder':HuggingFaceEmbedder,
    }