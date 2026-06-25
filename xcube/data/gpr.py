# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import torch
from loguru import logger

from xcube.data.base import DatasetSpec as DS
from xcube.data.base import RandomSafeDataset

import fvdb
# Original: fvdb._Cpp.SparseGridBatch = fvdb._Cpp.GridBatch
# Guarded because fvdb_core 0.4.2 no longer exposes a `_Cpp` submodule (GridBatch lives at fvdb.GridBatch directly).
if hasattr(fvdb, "_Cpp"):
    fvdb._Cpp.SparseGridBatch = fvdb._Cpp.GridBatch

import pickle
custom_pickle = pickle
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "featurevdb._Cpp":
            module = "fvdb._Cpp"
        return super().find_class(module, name)
custom_pickle.Unpickler = CustomUnpickler


class GPRDataset(RandomSafeDataset):
    """
    Pairs a TEUNet sparse reconstruction (input_grid) with its ground-truth
    sparse occupancy (target_grid), produced by preprocess_gpr.py.
    """

    def __init__(self, base_path, split, resolution, spec=None,
                 random_seed=0, hparams=None, skip_on_error=False,
                 custom_name="gpr", duplicate_num=1, input_key="input_grid", **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)
        self.skip_on_error = skip_on_error
        self.custom_name = custom_name
        self.resolution = resolution
        self.split = split
        self.spec = spec if spec is not None else [DS.INPUT_PC, DS.GT_DENSE_PC]
        # Stage 1 (VAE pretraining) sets input_key="target_grid" so the VAE
        # self-reconstructs GT shapes. Stage 2 (diffusion) uses the default
        # "input_grid" since TEUNet's output becomes a conditioning signal instead.
        self.input_key = input_key

        split_file = os.path.join(base_path, (split + '.lst'))
        with open(split_file, 'r') as f:
            stems = f.read().split('\n')
        if '' in stems:
            stems.remove('')
        self.all_items = [os.path.join(base_path, str(resolution), "%s.pkl" % s) for s in stems]

        logger.info(f"GPRDataset: {len(self.all_items)} items")
        self.hparams = hparams
        self.duplicate_num = duplicate_num

    def __len__(self):
        return len(self.all_items) * self.duplicate_num

    def get_name(self):
        return f"{self.custom_name}-{self.split}"

    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        item_path = self.all_items[data_id % len(self.all_items)]
        input_data = torch.load(item_path, pickle_module=custom_pickle)

        data = {}
        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = item_path

        if DS.INPUT_PC in self.spec:
            data[DS.INPUT_PC] = input_data[self.input_key]

        if DS.GT_DENSE_PC in self.spec:
            data[DS.GT_DENSE_PC] = input_data['target_grid']

        if DS.INPUT_INTENSITY in self.spec:
            data[DS.INPUT_INTENSITY] = input_data['input_prob']

        # Stage 2 (diffusion) needs both grids at once: the GT shape (as INPUT_PC,
        # via input_key="target_grid" in the diffusion config) to learn/denoise, and
        # TEUNet's flawed grid as a separate conditioning hint -- always from
        # 'input_grid' regardless of self.input_key, since that's specifically
        # TEUNet's reconstruction in the saved .pkl.
        if DS.COND_PC in self.spec:
            data[DS.COND_PC] = input_data['input_grid']

        return data
