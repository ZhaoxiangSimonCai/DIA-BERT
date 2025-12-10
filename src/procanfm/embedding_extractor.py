"""
DIA-BERT Embedding Extraction for ProCanFM Integration.

This module extracts latent representations from DIA-BERT for use in
the ProCanFM multimodal cancer foundation model. Embeddings are extracted
at the precursor level and then aggregated to sample level for fusion
with histopathology (TITAN/CONCH) and protein abundance embeddings.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.common.model.score_model import DIABertModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DIABERTEmbeddingExtractor:
    """Extract and aggregate embeddings from DIA-BERT for ProCanFM."""

    # Supported embedding dimensions
    EMBEDDING_DIMS = {
        'final': 64,
        'penultimate': 256,
    }

    def __init__(self, checkpoint_path: str, device: str = 'cuda', embedding_layer: str = 'penultimate'):
        """
        Initialize the embedding extractor.

        Args:
            checkpoint_path: Path to finetuned DIA-BERT checkpoint (.ckpt)
            device: Device to run inference on ('cuda' or 'cpu')
            embedding_layer: Which layer to extract from:
                - 'final': 64-dim (more compressed)
                - 'penultimate': 256-dim (richer, default, recommended for foundation models)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.embedding_layer = embedding_layer
        self.embedding_dim = self.EMBEDDING_DIMS.get(embedding_layer, 256)
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

    def _load_model(self, checkpoint_path: str) -> DIABertModel:
        """Load DIABertModel from checkpoint."""
        logger.info(f"Loading model from {checkpoint_path}")
        model = DIABertModel.load(checkpoint_path)
        logger.info("Model loaded successfully")
        return model

    @torch.no_grad()
    def extract_batch(
        self,
        rsm: torch.Tensor,
        frag_info: torch.Tensor,
        feat: torch.Tensor,
    ) -> np.ndarray:
        """
        Extract embeddings for a single batch.

        Args:
            rsm: Peak group matrix tensor of shape (batch, 8, 72, 16)
            frag_info: Fragment info tensor of shape (batch, 72, 4)
            feat: Precursor features tensor of shape (batch, 10)

        Returns:
            numpy array of shape (batch_size, 64)
        """
        rsm = rsm.to(self.device)
        frag_info = frag_info.to(self.device)
        feat = feat.to(self.device)

        embeddings = self.model.get_embeddings(rsm, frag_info, feat, self.embedding_layer)
        return embeddings.cpu().numpy()

    @torch.no_grad()
    def extract_precursor_embeddings(
        self,
        dataloader,
        precursor_id_key: str = 'precursor_id',
        file_key: str = 'file',
    ) -> tuple[np.ndarray, list, list]:
        """
        Extract embeddings for all precursors in a dataloader.

        Args:
            dataloader: DataLoader yielding batches with rsm, frag_info, feat,
                        precursor_id, and file (sample_id) fields
            precursor_id_key: Key for precursor ID in batch dict
            file_key: Key for sample/file ID in batch dict

        Returns:
            embeddings: (N_precursors, 64) array
            precursor_ids: List of precursor identifiers
            sample_ids: List of sample/file identifiers for each precursor
        """
        all_embeddings = []
        all_precursor_ids = []
        all_sample_ids = []

        for batch in tqdm(dataloader, desc="Extracting precursor embeddings"):
            # Handle dict-style batch (from DIA-BERT dataset)
            if isinstance(batch, dict):
                rsm = batch['rsm']
                frag_info = batch['frag_info']
                feat = batch['feat']
                precursor_ids = batch[precursor_id_key]
                sample_ids = batch[file_key]
            # Handle tuple-style batch
            elif isinstance(batch, (list, tuple)):
                rsm, frag_info, feat, label, sample_ids, precursor_ids = batch
            else:
                raise ValueError(f"Unsupported batch type: {type(batch)}")

            # Extract embeddings
            embeddings = self.extract_batch(rsm, frag_info, feat)

            all_embeddings.append(embeddings)

            # Handle tensor or list precursor/sample IDs
            if isinstance(precursor_ids, torch.Tensor):
                precursor_ids = precursor_ids.cpu().tolist()
            if isinstance(sample_ids, torch.Tensor):
                sample_ids = sample_ids.cpu().tolist()

            all_precursor_ids.extend(precursor_ids)
            all_sample_ids.extend(sample_ids)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(f"Extracted {len(all_precursor_ids)} precursor embeddings "
                    f"with dimension {self.embedding_dim}")

        return all_embeddings, all_precursor_ids, all_sample_ids

    def aggregate_to_sample_level(
        self,
        precursor_embeddings: np.ndarray,
        sample_ids: list,
        method: Literal['mean', 'max', 'weighted_mean'] = 'mean',
    ) -> dict[str, np.ndarray]:
        """
        Aggregate precursor-level embeddings to sample-level.

        This is necessary because DIA-BERT operates on precursors, but
        ProCanFM multimodal fusion requires sample-level embeddings to
        align with histopathology (one embedding per slide/sample).

        Args:
            precursor_embeddings: (N_precursors, 64) array
            sample_ids: Sample ID for each precursor
            method: Aggregation method
                - 'mean': Simple average (good baseline)
                - 'max': Captures strongest signals
                - 'weighted_mean': Weights by embedding L2 norm

        Returns:
            Dictionary mapping sample_id -> sample_embedding (64,)
        """
        # Group embeddings by sample
        sample_groups: dict[str, list[np.ndarray]] = defaultdict(list)
        for emb, sid in zip(precursor_embeddings, sample_ids):
            sample_groups[sid].append(emb)

        # Aggregate
        sample_embeddings = {}
        for sid, embs in sample_groups.items():
            embs_array = np.stack(embs, axis=0)  # (n_precursors, 64)

            if method == 'mean':
                sample_embeddings[sid] = np.mean(embs_array, axis=0)
            elif method == 'max':
                sample_embeddings[sid] = np.max(embs_array, axis=0)
            elif method == 'weighted_mean':
                # Weight by L2 norm (higher magnitude = more confident)
                weights = np.linalg.norm(embs_array, axis=1, keepdims=True)
                weights = weights / (weights.sum() + 1e-8)
                sample_embeddings[sid] = (embs_array * weights).sum(axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

        logger.info(f"Aggregated {len(precursor_embeddings)} precursors into "
                    f"{len(sample_embeddings)} sample embeddings using '{method}'")

        return sample_embeddings

    def extract_and_save(
        self,
        dataloader,
        output_path: str,
        aggregation_method: Literal['mean', 'max', 'weighted_mean'] = 'mean',
        precursor_id_key: str = 'precursor_id',
        file_key: str = 'file',
    ) -> Path:
        """
        Full pipeline: extract precursor embeddings, aggregate to sample level,
        and save to HDF5 for ProCanFM training.

        Args:
            dataloader: DataLoader yielding batches of precursors
            output_path: Path to save HDF5 file
            aggregation_method: How to aggregate precursor -> sample embeddings
            precursor_id_key: Key for precursor ID in batch dict
            file_key: Key for sample/file ID in batch dict

        Returns:
            Path to the saved HDF5 file
        """
        # Extract precursor-level embeddings
        precursor_embs, precursor_ids, sample_ids = self.extract_precursor_embeddings(
            dataloader,
            precursor_id_key=precursor_id_key,
            file_key=file_key,
        )

        # Aggregate to sample level
        sample_embeddings = self.aggregate_to_sample_level(
            precursor_embs, sample_ids, method=aggregation_method
        )

        # Convert to arrays for storage
        sample_ids_sorted = sorted(sample_embeddings.keys())
        embeddings_array = np.stack(
            [sample_embeddings[sid] for sid in sample_ids_sorted]
        )

        # Save to HDF5
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            # Sample-level embeddings (for ProCanFM fusion)
            f.create_dataset(
                'sample_embeddings',
                data=embeddings_array,
                dtype='float32',
            )
            # Store sample IDs as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            sample_ids_ds = f.create_dataset('sample_ids', (len(sample_ids_sorted),), dtype=dt)
            for i, sid in enumerate(sample_ids_sorted):
                sample_ids_ds[i] = str(sid)

            # Also store precursor-level for potential fine-grained analysis
            f.create_dataset(
                'precursor_embeddings',
                data=precursor_embs,
                dtype='float32',
            )
            precursor_ids_ds = f.create_dataset('precursor_ids', (len(precursor_ids),), dtype=dt)
            for i, pid in enumerate(precursor_ids):
                precursor_ids_ds[i] = str(pid)

            precursor_sample_ds = f.create_dataset('precursor_sample_ids', (len(sample_ids),), dtype=dt)
            for i, sid in enumerate(sample_ids):
                precursor_sample_ds[i] = str(sid)

            # Metadata
            f.attrs['embedding_dim'] = self.embedding_dim
            f.attrs['n_samples'] = len(sample_ids_sorted)
            f.attrs['n_precursors'] = len(precursor_ids)
            f.attrs['aggregation_method'] = aggregation_method
            f.attrs['model_type'] = 'DIA-BERT'

        logger.info(f"Saved embeddings to {output_path}")
        logger.info(f"  - {len(sample_ids_sorted)} samples")
        logger.info(f"  - {len(precursor_ids)} precursors")
        logger.info(f"  - Embedding dimension: {self.embedding_dim}")

        return output_path


def load_embeddings(h5_path: str) -> dict:
    """
    Load embeddings from HDF5 file created by DIABERTEmbeddingExtractor.

    Args:
        h5_path: Path to HDF5 file

    Returns:
        Dictionary containing:
            - sample_embeddings: (N_samples, 64) array
            - sample_ids: list of sample IDs
            - precursor_embeddings: (N_precursors, 64) array
            - precursor_ids: list of precursor IDs
            - precursor_sample_ids: list mapping precursor -> sample
            - metadata: dict with embedding_dim, n_samples, etc.
    """
    with h5py.File(h5_path, 'r') as f:
        result = {
            'sample_embeddings': f['sample_embeddings'][:],
            'sample_ids': [s.decode() if isinstance(s, bytes) else s for s in f['sample_ids'][:]],
            'precursor_embeddings': f['precursor_embeddings'][:],
            'precursor_ids': [s.decode() if isinstance(s, bytes) else s for s in f['precursor_ids'][:]],
            'precursor_sample_ids': [s.decode() if isinstance(s, bytes) else s for s in f['precursor_sample_ids'][:]],
            'metadata': dict(f.attrs),
        }
    return result


def main():
    """CLI for batch embedding extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract DIA-BERT embeddings for ProCanFM multimodal fusion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python -m src.procanfm.embedding_extractor \\
        --checkpoint resource/model/finetune_model.ckpt \\
        --output embeddings/diabert_embeddings.h5 \\
        --aggregation mean \\
        --device cuda

Note: This script requires a DataLoader to be set up with your DIA data.
See the DIABERTEmbeddingExtractor class for programmatic usage.
        """
    )
    parser.add_argument(
        '--checkpoint', required=True,
        help='Path to DIA-BERT checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output HDF5 path for embeddings'
    )
    parser.add_argument(
        '--aggregation', default='mean',
        choices=['mean', 'max', 'weighted_mean'],
        help='Sample-level aggregation method (default: mean)'
    )
    parser.add_argument(
        '--device', default='cuda',
        help='Device for inference (cuda or cpu, default: cuda)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='Batch size for extraction (default: 256)'
    )

    args = parser.parse_args()

    # Initialize extractor
    extractor = DIABERTEmbeddingExtractor(args.checkpoint, args.device)

    print("\n" + "=" * 60)
    print("DIA-BERT Embedding Extractor for ProCanFM")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Device: {extractor.device}")
    print(f"Embedding dim: {extractor.embedding_dim}")
    print("=" * 60 + "\n")

    # Placeholder for dataloader setup
    # Users need to implement their own dataloader based on their data format
    print("To extract embeddings, create a DataLoader with your DIA data and call:")
    print("    extractor.extract_and_save(dataloader, args.output, args.aggregation)")
    print("\nExample programmatic usage:")
    print("""
    from src.procanfm.embedding_extractor import DIABERTEmbeddingExtractor
    from torch.utils.data import DataLoader
    
    # Load your dataset
    dataset = YourDIADataset(...)  # Must return dict with rsm, frag_info, feat, precursor_id, file
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Extract embeddings
    extractor = DIABERTEmbeddingExtractor('path/to/checkpoint.ckpt')
    extractor.extract_and_save(dataloader, 'embeddings.h5', aggregation_method='mean')
    
    # Load embeddings for ProCanFM
    from src.procanfm.embedding_extractor import load_embeddings
    data = load_embeddings('embeddings.h5')
    sample_embeddings = data['sample_embeddings']  # Shape: (N_samples, 64)
    """)


if __name__ == '__main__':
    main()

