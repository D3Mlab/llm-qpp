from .embedders import RandomEmbedder, OpenAIEmbedder, HuggingFaceEmbedder, TestQueryEmbedder

EMBEDDER_CLASSES = {
    'HuggingFaceEmbedder': HuggingFaceEmbedder,
    'RandomEmbedder':RandomEmbedder,
    'HuggingFaceEmbedder':HuggingFaceEmbedder,
    'TestQueryEmbedder': TestQueryEmbedder
    }