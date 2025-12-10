"""
Embedding Extraction Process for DIA-BERT Pipeline Integration.

This module provides a process class that integrates with the DIA-BERT
identification pipeline to automatically extract embeddings for ProCanFM.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.common.model.score_model import DIABertModel
from src.utils import msg_send_utils

logger = logging.getLogger(__name__)


def collate_batch(batch):
    """Collate function for DataLoader."""
    rsm = torch.tensor(np.array([x['rsm'] for x in batch]), dtype=torch.float32)
    frag_info = torch.tensor(np.array([x['frag_info'] for x in batch]), dtype=torch.float32)
    feat = torch.tensor(np.array([x['feat'] for x in batch]), dtype=torch.float32)
    label = torch.tensor(np.array([x['label'] for x in batch]), dtype=torch.float32)
    file_name = [x['file'] for x in batch]
    precursor_id = [x['precursor_id'] for x in batch]
    return rsm, frag_info, feat, label, file_name, precursor_id


class EmbeddingExtractionProcess:
    """
    Process class for extracting DIA-BERT embeddings as part of the pipeline.
    
    This class loads the finetuned model and extracts embeddings for all
    identified precursors, then aggregates to sample level for ProCanFM.
    """

    # Embedding dimensions for each layer
    EMBEDDING_DIMS = {
        'final': 64,
        'penultimate': 256,
    }

    def __init__(
        self,
        base_output: str,
        rawdata_prefix: str,
        mzml_name: str,
        model_path: str,
        device: str = 'cuda',
        batch_size: int = 256,
        aggregation_method: str = 'mean',
        embedding_layer: str = 'penultimate',
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the embedding extraction process.

        Args:
            base_output: Base output directory for the mzML file
            rawdata_prefix: Prefix of the raw data file (mzML name without extension)
            mzml_name: Full mzML filename
            model_path: Path to the model checkpoint (finetuned or base)
            device: Device to run inference on
            batch_size: Batch size for embedding extraction
            aggregation_method: How to aggregate precursor embeddings to sample level
            embedding_layer: Which layer to extract embeddings from:
                - 'final': 64-dim (more compressed)
                - 'penultimate': 256-dim (richer, recommended for foundation models)
            logger_instance: Logger instance for output
        """
        self.base_output = base_output
        self.rawdata_prefix = rawdata_prefix
        self.mzml_name = mzml_name
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.aggregation_method = aggregation_method
        self.embedding_layer = embedding_layer
        self.embedding_dim = self.EMBEDDING_DIMS.get(embedding_layer, 256)
        self.logger = logger_instance or logger

    def deal_process(self) -> Optional[str]:
        """
        Main process to extract embeddings.

        Returns:
            Path to the saved embeddings HDF5 file, or None if extraction failed
        """
        try:
            msg_send_utils.send_msg(msg='Extracting DIA-BERT embeddings for ProCanFM')
            self.logger.info('Starting embedding extraction for ProCanFM')

            # Load the model
            model = self._load_model()
            if model is None:
                self.logger.warning('Could not load model, skipping embedding extraction')
                return None

            # Load precursor data
            dataloader = self._create_dataloader()
            if dataloader is None:
                self.logger.warning('No data available for embedding extraction')
                return None

            # Load FDR-filtered precursor IDs to filter embeddings
            fdr_precursor_ids = self._load_fdr_filtered_precursors()

            # Extract embeddings
            precursor_embeddings, precursor_ids, sample_ids = self._extract_embeddings(
                model, dataloader, fdr_precursor_ids
            )

            if len(precursor_embeddings) == 0:
                self.logger.warning('No embeddings extracted')
                return None

            # Aggregate to sample level
            sample_embeddings = self._aggregate_to_sample_level(
                precursor_embeddings, sample_ids
            )

            # Save to HDF5
            output_path = self._save_embeddings(
                precursor_embeddings, precursor_ids, sample_ids, sample_embeddings
            )

            self.logger.info(f'Embedding extraction complete: {output_path}')
            msg_send_utils.send_msg(msg='DIA-BERT embedding extraction complete')

            return output_path

        except Exception as e:
            self.logger.exception(f'Embedding extraction failed: {e}')
            return None

    def _load_model(self) -> Optional[DIABertModel]:
        """Load the DIA-BERT model from checkpoint."""
        # First try finetuned model
        finetuned_model_dir = os.path.join(self.base_output, 'finetune', 'model')
        finetuned_model_path = None

        if os.path.exists(finetuned_model_dir):
            # Find the latest finetuned model
            model_files = [f for f in os.listdir(finetuned_model_dir) if f.endswith('.ckpt')]
            if model_files:
                # Sort by epoch number to get the latest
                model_files.sort(key=lambda x: int(x.split('epoch=')[1].split('-')[0]) if 'epoch=' in x else 0)
                finetuned_model_path = os.path.join(finetuned_model_dir, model_files[-1])

        # Use finetuned model if available, otherwise use provided model path
        model_path = finetuned_model_path if finetuned_model_path else self.model_path

        if not os.path.exists(model_path):
            self.logger.error(f'Model file not found: {model_path}')
            return None

        self.logger.info(f'Loading model from: {model_path}')
        model = DIABertModel.load(model_path)
        model.to(self.device)
        model.eval()
        return model

    def _create_dataloader(self) -> Optional[DataLoader]:
        """Create dataloader from cached finetune data."""
        finetune_data_path = os.path.join(self.base_output, 'finetune', 'data')

        if not os.path.exists(finetune_data_path):
            self.logger.warning(f'Finetune data path not found: {finetune_data_path}')
            return None

        # Find all chunk pickle files
        pkl_files = [
            os.path.join(finetune_data_path, f)
            for f in os.listdir(finetune_data_path)
            if f.endswith('.pkl') and f.startswith('chunk_')
        ]

        if not pkl_files:
            self.logger.warning('No chunk pickle files found')
            return None

        self.logger.info(f'Loading {len(pkl_files)} chunk files for embedding extraction')

        # Load all datasets
        datasets = []
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    dataset = pickle.load(f)
                    datasets.append(dataset)
            except Exception as e:
                self.logger.warning(f'Failed to load {pkl_file}: {e}')
                continue

        if not datasets:
            return None

        # Combine datasets
        combined_dataset = ConcatDataset(datasets)

        # Create dataloader
        dataloader = DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=collate_batch,
        )

        return dataloader

    def _load_fdr_filtered_precursors(self) -> Optional[set]:
        """Load precursor IDs that passed FDR filtering."""
        # Try to load from final precursor results
        precursor_csv = os.path.join(
            self.base_output, f'{self.rawdata_prefix}_precursor.csv'
        )

        if os.path.exists(precursor_csv):
            try:
                df = pd.read_csv(precursor_csv)
                if 'Precursor' in df.columns:
                    return set(df['Precursor'].tolist())
            except Exception as e:
                self.logger.warning(f'Could not load precursor CSV: {e}')

        # Try to load from FDR file
        fdr_csv = os.path.join(
            self.base_output, 'finetune', 'output', f'fdr_{self.mzml_name}_eval.csv'
        )

        if os.path.exists(fdr_csv):
            try:
                df = pd.read_csv(fdr_csv)
                # Filter to targets with q_value <= 0.01
                if 'q_value' in df.columns and 'label' in df.columns:
                    filtered = df[(df['q_value'] <= 0.01) & (df['label'] == 1)]
                    if 'transition_group_id' in filtered.columns:
                        return set(filtered['transition_group_id'].tolist())
            except Exception as e:
                self.logger.warning(f'Could not load FDR CSV: {e}')

        # If no filter available, return None to use all precursors
        self.logger.info('No FDR filter found, extracting embeddings for all precursors')
        return None

    @torch.no_grad()
    def _extract_embeddings(
        self,
        model: DIABertModel,
        dataloader: DataLoader,
        fdr_precursor_ids: Optional[set],
    ) -> tuple[np.ndarray, list, list]:
        """Extract embeddings for all precursors."""
        all_embeddings = []
        all_precursor_ids = []
        all_sample_ids = []

        for batch in dataloader:
            rsm, frag_info, feat, label, file_names, precursor_ids = batch

            # Move to device
            rsm = rsm.to(self.device)
            frag_info = frag_info.to(self.device)
            feat = feat.to(self.device)

            # Extract embeddings from specified layer
            embeddings = model.get_embeddings(rsm, frag_info, feat, self.embedding_layer)
            embeddings = embeddings.cpu().numpy()

            # Filter by FDR if available
            for i, (emb, pid, fname) in enumerate(zip(embeddings, precursor_ids, file_names)):
                if fdr_precursor_ids is None or pid in fdr_precursor_ids:
                    all_embeddings.append(emb)
                    all_precursor_ids.append(pid)
                    all_sample_ids.append(fname)

        if all_embeddings:
            all_embeddings = np.stack(all_embeddings, axis=0)
        else:
            all_embeddings = np.array([])

        self.logger.info(
            f'Extracted {len(all_precursor_ids)} precursor embeddings '
            f'(layer={self.embedding_layer}, dim={self.embedding_dim})'
        )

        return all_embeddings, all_precursor_ids, all_sample_ids

    def _aggregate_to_sample_level(
        self,
        precursor_embeddings: np.ndarray,
        sample_ids: list,
    ) -> dict[str, np.ndarray]:
        """Aggregate precursor embeddings to sample level."""
        if len(precursor_embeddings) == 0:
            return {}

        # Group by sample
        sample_groups: dict[str, list[np.ndarray]] = defaultdict(list)
        for emb, sid in zip(precursor_embeddings, sample_ids):
            sample_groups[sid].append(emb)

        # Aggregate
        sample_embeddings = {}
        for sid, embs in sample_groups.items():
            embs_array = np.stack(embs, axis=0)

            if self.aggregation_method == 'mean':
                sample_embeddings[sid] = np.mean(embs_array, axis=0)
            elif self.aggregation_method == 'max':
                sample_embeddings[sid] = np.max(embs_array, axis=0)
            elif self.aggregation_method == 'weighted_mean':
                weights = np.linalg.norm(embs_array, axis=1, keepdims=True)
                weights = weights / (weights.sum() + 1e-8)
                sample_embeddings[sid] = (embs_array * weights).sum(axis=0)
            else:
                sample_embeddings[sid] = np.mean(embs_array, axis=0)

        self.logger.info(
            f'Aggregated to {len(sample_embeddings)} sample embeddings '
            f'using {self.aggregation_method}'
        )

        return sample_embeddings

    def _save_embeddings(
        self,
        precursor_embeddings: np.ndarray,
        precursor_ids: list,
        sample_ids: list,
        sample_embeddings: dict[str, np.ndarray],
    ) -> str:
        """Save embeddings to HDF5 file."""
        output_path = os.path.join(
            self.base_output, f'{self.rawdata_prefix}_embeddings.h5'
        )

        # Convert sample embeddings to arrays
        sample_ids_sorted = sorted(sample_embeddings.keys())
        sample_emb_array = np.stack(
            [sample_embeddings[sid] for sid in sample_ids_sorted]
        ) if sample_embeddings else np.array([])

        with h5py.File(output_path, 'w') as f:
            # Sample-level embeddings (for ProCanFM fusion)
            f.create_dataset('sample_embeddings', data=sample_emb_array, dtype='float32')

            # Store sample IDs
            dt = h5py.special_dtype(vlen=str)
            if sample_ids_sorted:
                sample_ids_ds = f.create_dataset('sample_ids', (len(sample_ids_sorted),), dtype=dt)
                for i, sid in enumerate(sample_ids_sorted):
                    sample_ids_ds[i] = str(sid)

            # Precursor-level embeddings
            if len(precursor_embeddings) > 0:
                f.create_dataset('precursor_embeddings', data=precursor_embeddings, dtype='float32')

                precursor_ids_ds = f.create_dataset('precursor_ids', (len(precursor_ids),), dtype=dt)
                for i, pid in enumerate(precursor_ids):
                    precursor_ids_ds[i] = str(pid)

                precursor_sample_ds = f.create_dataset('precursor_sample_ids', (len(sample_ids),), dtype=dt)
                for i, sid in enumerate(sample_ids):
                    precursor_sample_ds[i] = str(sid)

            # Metadata
            f.attrs['embedding_dim'] = self.embedding_dim
            f.attrs['embedding_layer'] = self.embedding_layer
            f.attrs['n_samples'] = len(sample_ids_sorted)
            f.attrs['n_precursors'] = len(precursor_ids)
            f.attrs['aggregation_method'] = self.aggregation_method
            f.attrs['model_type'] = 'DIA-BERT'
            f.attrs['mzml_name'] = self.mzml_name
            f.attrs['rawdata_prefix'] = self.rawdata_prefix

        self.logger.info(f'Saved embeddings to {output_path}')
        self.logger.info(f'  - {len(sample_ids_sorted)} sample(s)')
        self.logger.info(f'  - {len(precursor_ids)} precursors')
        self.logger.info(f'  - Embedding layer: {self.embedding_layer} ({self.embedding_dim}-dim)')

        return output_path

