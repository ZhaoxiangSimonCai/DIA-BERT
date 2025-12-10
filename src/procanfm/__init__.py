"""
ProCanFM Integration Module for DIA-BERT.

This module provides utilities for extracting latent embeddings from DIA-BERT
for use in the ProCanFM multimodal cancer foundation model.
"""

from src.procanfm.embedding_extractor import DIABERTEmbeddingExtractor
from src.procanfm.embedding_extraction_process import EmbeddingExtractionProcess

__all__ = ['DIABERTEmbeddingExtractor', 'EmbeddingExtractionProcess']

