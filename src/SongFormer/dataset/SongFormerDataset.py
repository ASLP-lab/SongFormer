# For this dataset, ablation studies become easier
import copy
import json
import pdb
from argparse import Namespace
from pathlib import Path
import traceback
import numpy as np
import torch
from dataset.custom_types import MsaInfo
from dataset.label2id import (
    DATASET_ID_ALLOWED_LABEL_IDS,
    DATASET_LABEL_TO_DATASET_ID,
    ID_TO_LABEL,
    LABEL_TO_ID,
)
from loguru import logger
from scipy.ndimage import maximum_filter1d
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import os
import random
from .HookTheoryAdapter import HookTheoryAdapter
from .GeminiOnlyLabelAdapter import GeminiOnlyLabelAdapter


class Dataset(Dataset):
    def get_ids_from_dir(self, dir_path: str):
        ids = os.listdir(dir_path)
        ids = [Path(x).stem for x in ids if x.endswith(".npy")]
        return set(ids)

    def __init__(
        self,
        dataset_abstracts: dict,
        hparams,
    ):
        # initialize storage and hyperparams
        self.time_datas = {}
        self.label_datas = {}
        self.hparams = hparams
        self.label_to_id = LABEL_TO_ID
        self.dataset_id_to_dataset_id = DATASET_LABEL_TO_DATASET_ID
        self.id_to_label = ID_TO_LABEL
        self.dataset_id2label_mask = {}
        self.output_logits_frame_rates = self.hparams.output_logits_frame_rates
        self.downsample_rates = self.hparams.downsample_rates
        self.valid_data_ids = []
        self.SLICE_DUR = self.hparams.slice_dur

        self.input_embedding_dir = {}
        self.EPS = 1e-6

        # build dataset-specific label mask
        for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
            self.dataset_id2label_mask[key] = np.ones(
                self.hparams.num_classes, dtype=bool
            )
            self.dataset_id2label_mask[key][allowed_ids] = False

        uniq_id_nums = 0
        self.adapter_obj = {}

        for dataset_abstract_item in dataset_abstracts:
            adapter = dataset_abstract_item.get("adapter", None)
            if adapter is not None:
                # adapter-based dataset (pre-wrapped)
                assert isinstance(adapter, str)
                if adapter == "HookTheoryAdapter":
                    self.adapter_obj[dataset_abstract_item["internal_tmp_id"]] = (
                        HookTheoryAdapter(**dataset_abstract_item, hparams=self.hparams)
                    )
                    valid_data_ids = self.adapter_obj[
                        dataset_abstract_item["internal_tmp_id"]
                    ].get_ids()
                elif adapter == "GeminiOnlyLabelAdapter":
                    self.adapter_obj[dataset_abstract_item["internal_tmp_id"]] = (
                        GeminiOnlyLabelAdapter(
                            **dataset_abstract_item, hparams=self.hparams
                        )
                    )
                    valid_data_ids = self.adapter_obj[
                        dataset_abstract_item["internal_tmp_id"]
                    ].get_ids()
                else:
                    raise ValueError(f"Unknown adapter: {adapter}")

                logger.info(
                    f"{dataset_abstract_item['internal_tmp_id']}: {len(valid_data_ids)} * {dataset_abstract_item['multiplier']}"
                )
                uniq_id_nums += len(valid_data_ids)
                for i in range(dataset_abstract_item["multiplier"]):
                    self.valid_data_ids.extend(valid_data_ids)

            else:
                # raw dataset definition
                internal_tmp_id = dataset_abstract_item["internal_tmp_id"]
                dataset_type = dataset_abstract_item["dataset_type"]
                all_input_embedding_dirs = dataset_abstract_item[
                    "input_embedding_dir"
                ].split()
                label_path = dataset_abstract_item["label_path"]
                split_ids_path = dataset_abstract_item["split_ids_path"]

                self.input_embedding_dir[internal_tmp_id] = dataset_abstract_item[
                    "input_embedding_dir"
                ]

                valid_data_ids = self.get_ids_from_dir(all_input_embedding_dirs[0])
                for x in all_input_embedding_dirs:
                    valid_data_ids = valid_data_ids.intersection(
                        self.get_ids_from_dir(x)
                    )

                split_ids = []
                with open(split_ids_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        split_ids.append(line.strip())
                split_ids = set(split_ids)

                # filter valid ids by split membership
                valid_data_ids = [
                    x
                    for x in valid_data_ids
                    if "_".join(x.split("_")[:-1]) in split_ids
                ]

                valid_data_ids = [
                    (internal_tmp_id, dataset_type, x, None) for x in valid_data_ids
                ]

                assert isinstance(dataset_abstract_item["multiplier"], int)
                uniq_id_nums += len(valid_data_ids)
                logger.info(
                    f"{internal_tmp_id}: {len(valid_data_ids)} * {dataset_abstract_item['multiplier']}"
                )
                for i in range(dataset_abstract_item["multiplier"]):
                    self.valid_data_ids.extend(valid_data_ids)
                self.init_segments(
                    label_path=label_path, internal_tmp_id=internal_tmp_id
                )

        logger.info(f"{uniq_id_nums} valid data ids, {len(self.valid_data_ids)} total")
        rng = np.random.default_rng(42)
        rng.shuffle(self.valid_data_ids)

    def init_segments(
        self,
        label_path,
        internal_tmp_id,
    ):
        # load segment times and labels from label jsonl
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if not line:
                    continue
                line_data = json.loads(line)
                hybrid_id = internal_tmp_id + "_" + line_data["id"]
                self.time_datas[hybrid_id] = [x[0] for x in line_data["labels"]]
                self.time_datas[hybrid_id] = list(
                    map(float, self.time_datas[hybrid_id])
                )
                self.label_datas[hybrid_id] = [
                    -1 if x[1] == "end" else self.label_to_id[x[1]]
                    for x in line_data["labels"]
                ]

    def __len__(self):
        return len(self.valid_data_ids)

    def widen_temporal_events(self, events, num_neighbors):
        # smooth discrete events with normalized Gaussian kernel
        def theoretical_gaussian_max(sigma):
            return 1 / (np.sqrt(2 * np.pi) * sigma)

        widen_events = events
        sigma = num_neighbors / 3.0
        smoothed = gaussian_filter1d(widen_events.astype(float), sigma=sigma)
        smoothed /= theoretical_gaussian_max(sigma)
        smoothed = np.clip(smoothed, 0, 1)

        return smoothed

    def time2frame(self, this_time):
        assert this_time <= self.SLICE_DUR
        return int(this_time * self.output_logits_frame_rates)

    def __getitem__(self, idx):
        try:
            internal_tmp_id, dataset_label, utt, adapter_str = self.valid_data_ids[idx]
            if adapter_str is not None:
                # handle adapter-wrapped entries
                assert isinstance(adapter_str, str)
                if adapter_str == "HookTheoryAdapter":
                    start_time = int(utt.split("_")[-1])
                    return self.adapter_obj[internal_tmp_id].get_item_json(
                        utt=utt,
                        start_time=start_time,
                        end_time=start_time + self.SLICE_DUR,
                    )
                elif adapter_str == "GeminiOnlyLabelAdapter":
                    start_time = int(utt.split("_")[-1])
                    return self.adapter_obj[internal_tmp_id].get_item_json(
                        utt=utt,
                        start_time=start_time,
                        end_time=start_time + self.SLICE_DUR,
                    )
                else:
                    raise ValueError(f"Unknown adapter: {adapter_str}")

            # load embeddings from configured dirs
            embd_list = []
            embd_dirs = self.input_embedding_dir[internal_tmp_id].split()
            for embd_dir in embd_dirs:
                if not Path(embd_dir).exists():
                    raise FileNotFoundError(
                        f"Embedding directory {embd_dir} does not exist"
                    )
                tmp = np.load(Path(embd_dir) / f"{utt}.npy").squeeze(axis=0)
                embd_list.append(tmp)

            # check that max/min length difference across embeddings <= 4
            if len(embd_list) > 1:
                embd_shapes = [x.shape for x in embd_list]
                max_shape = max(embd_shapes, key=lambda x: x[0])
                min_shape = min(embd_shapes, key=lambda x: x[0])
                if abs(max_shape[0] - min_shape[0]) > 4:
                    raise ValueError(
                        f"Embedding shapes differ too much: {max_shape} vs {min_shape}"
                    )
            if len(embd_list) > 1:
                for idx in range(len(embd_list)):
                    embd_list[idx] = embd_list[idx][: min_shape[0], :]

            input_embedding = np.concatenate(embd_list, axis=-1)

            start_time = int(utt.split("_")[-1])
            utt_id_with_start_sec = utt
            utt = "_".join(utt.split("_")[:-1])
            end_time = start_time + self.SLICE_DUR

            local_times = np.array(
                copy.deepcopy(self.time_datas[f"{internal_tmp_id}_{utt}"])
            )
            local_labels = copy.deepcopy(self.label_datas[f"{internal_tmp_id}_{utt}"])

            assert np.all(local_times[:-1] < local_times[1:]), (
                f"time must be sorted, but {utt} is {local_times}"
            )

            local_times = local_times - start_time

            time_L = max(0.0, float(local_times.min()))
            time_R = min(float(self.SLICE_DUR), float(local_times.max()))

            keep_boundarys = (time_L + self.EPS < local_times) & (
                local_times < time_R - self.EPS
            )

            # If no valid boundaries, return None (skip)
            if keep_boundarys.sum() <= 0:
                return None

            mask = np.ones(
                [int(self.SLICE_DUR * self.output_logits_frame_rates)], dtype=bool
            )
            mask[self.time2frame(time_L) : self.time2frame(time_R)] = False

            true_boundary = np.zeros(
                [int(self.SLICE_DUR * self.output_logits_frame_rates)], dtype=float
            )
            for idx in np.flatnonzero(keep_boundarys):
                true_boundary[self.time2frame(local_times[idx])] = 1

            true_function = np.zeros(
                [
                    int(self.SLICE_DUR * self.output_logits_frame_rates),
                    self.hparams.num_classes,
                ],
                dtype=float,
            )
            true_function_list = []
            msa_info: MsaInfo = []
            last_pos = self.time2frame(time_L)
            for idx in np.flatnonzero(keep_boundarys):
                true_function[
                    last_pos : self.time2frame(local_times[idx]),
                    int(local_labels[idx - 1]),
                ] = 1
                true_function_list.append(int(local_labels[idx - 1]))
                last_pos = self.time2frame(local_times[idx])
                msa_info.append(
                    (
                        float(max(local_times[idx - 1], time_L)),
                        str(self.id_to_label[int(local_labels[idx - 1])]),
                    )
                )

            true_function[
                last_pos : self.time2frame(time_R),
                local_labels[int(np.flatnonzero(keep_boundarys)[-1])],
            ] = 1
            true_function_list.append(
                int(local_labels[int(np.flatnonzero(keep_boundarys)[-1])])
            )
            msa_info.append(
                (
                    float(local_times[int(np.flatnonzero(keep_boundarys)[-1])]),
                    str(
                        self.id_to_label[
                            int(local_labels[int(np.flatnonzero(keep_boundarys)[-1])])
                        ]
                    ),
                )
            )
            msa_info.append((float(time_R), "end"))

            return {
                "data_id": internal_tmp_id + "_" + utt_id_with_start_sec,
                "input_embedding": input_embedding,
                "mask": mask,
                "true_boundary": true_boundary,
                "widen_true_boundary": self.widen_temporal_events(
                    true_boundary, num_neighbors=self.hparams.num_neighbors
                ),
                "true_function": true_function,
                "true_function_list": true_function_list,
                "msa_info": msa_info,
                "dataset_id": self.dataset_id_to_dataset_id[dataset_label],
                "label_id_mask": self.dataset_id2label_mask[
                    self.dataset_id_to_dataset_id[dataset_label]
                ],
            }
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(
                f"error in __getitem__, idx={idx}, utt={utt}, error is:\n{e}\n{tb_str}"
            )
            return None

    def collate_fn(self, batch):
        """
        Return dictionary including:
        - data_ids
        - input_embeddings
        - masks
        - true_boundaries
        - widen_true_boundaries
        - true_functions
        - true_function_lists
        """
        try:
            # filter out None entries
            batch = [x for x in batch if x is not None]
            if len(batch) == 0:
                return None

            data_ids = []
            max_embeddings_length = max([x["input_embedding"].shape[0] for x in batch])
            max_sequence_length = max_embeddings_length // self.downsample_rates

            # allocate numpy arrays for batch
            input_embeddings = np.zeros(
                (len(batch), max_embeddings_length, self.hparams.input_dim), dtype=float
            )
            masks = np.ones((len(batch), max_sequence_length), dtype=bool)
            true_boundaries = np.zeros((len(batch), max_sequence_length), dtype=float)
            widen_true_boundaries = np.zeros(
                (len(batch), max_sequence_length), dtype=float
            )
            true_functions = np.zeros(
                (len(batch), max_sequence_length, self.hparams.num_classes), dtype=float
            )
            boundary_mask = np.zeros((len(batch), max_sequence_length), dtype=bool)
            function_mask = np.zeros((len(batch), max_sequence_length), dtype=bool)
            true_function_lists = []
            msa_infos = []
            dataset_ids = []
            label_id_masks = []

            for idx, item in enumerate(batch):
                data_ids.append(item["data_id"])
                input_embeddings[idx, : item["input_embedding"].shape[0]] = item[
                    "input_embedding"
                ]
                masks[idx, : item["mask"].shape[0]] = item["mask"][:max_sequence_length]
                true_boundaries[idx, : item["true_boundary"].shape[0]] = item[
                    "true_boundary"
                ][:max_sequence_length]
                widen_true_boundaries[idx, : item["widen_true_boundary"].shape[0]] = (
                    item["widen_true_boundary"]
                )[:max_sequence_length]
                true_functions[idx, : item["true_function"].shape[0]] = item[
                    "true_function"
                ][:max_sequence_length]
                true_function_lists.append(item["true_function_list"])
                msa_infos.append(item["msa_info"])
                dataset_ids.append(item["dataset_id"])
                label_id_masks.append(item["label_id_mask"])
                if boundary_mask is not None:
                    boundary_mask[idx, : item["mask"].shape[0]] = item.get(
                        "boundary_mask", np.zeros(item["mask"].shape[0], dtype=bool)
                    )[:max_sequence_length]
                if function_mask is not None:
                    function_mask[idx, : item["mask"].shape[0]] = item.get(
                        "function_mask", np.zeros(item["mask"].shape[0], dtype=bool)
                    )[:max_sequence_length]

            # convert to torch tensors
            input_embeddings = torch.from_numpy(input_embeddings).float()
            masks = torch.from_numpy(masks).bool()
            true_boundaries = torch.from_numpy(true_boundaries).float()
            widen_true_boundaries = torch.from_numpy(widen_true_boundaries).float()
            true_functions = torch.from_numpy(true_functions).float()
            boundary_mask = torch.from_numpy(boundary_mask).bool()
            function_mask = torch.from_numpy(function_mask).bool()
            true_function_lists = [
                torch.tensor(x, dtype=torch.long) for x in true_function_lists
            ]
            dataset_ids = torch.from_numpy(np.array(dataset_ids, dtype=np.int64))

            label_id_masks = torch.from_numpy(
                np.stack(label_id_masks, axis=0, dtype=bool)[:, np.newaxis, :]
            )

            return_json = {
                "data_ids": data_ids,
                "input_embeddings": input_embeddings,
                "masks": masks,
                "true_boundaries": true_boundaries,
                "widen_true_boundaries": widen_true_boundaries,
                "true_functions": true_functions,
                "true_function_lists": true_function_lists,
                "msa_infos": msa_infos,
                "dataset_ids": dataset_ids,
                "label_id_masks": label_id_masks,
                "boundary_mask": boundary_mask,
                "function_mask": function_mask,
            }

            return return_json
        except Exception as e:
            logger.error(f"Error occurred while processing dataset: {e}")
            return None