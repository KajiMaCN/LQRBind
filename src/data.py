from __future__ import annotations

import ast
import csv
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


def set_seed(seed: int, deterministic: bool = False):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)


def _construct_primary_graph(sequence_length: int, window_size: int = 13) -> torch.Tensor:
    start_points, end_points = [], []
    for i in range(sequence_length):
        for j in range(max(0, i - window_size), min(sequence_length, i + window_size + 1)):
            if i != j:
                start_points.append(i)
                end_points.append(j)
    return torch.tensor([start_points, end_points], dtype=torch.long)


def _deduplicate_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for src, dst in edge_index.T.tolist():
        src_i, dst_i = int(src), int(dst)
        if src_i == dst_i or src_i < 0 or dst_i < 0 or src_i >= num_nodes or dst_i >= num_nodes:
            continue
        pair = (src_i, dst_i)
        if pair in seen:
            continue
        seen.add(pair)
        edges.append(pair)
    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).T.contiguous()


def _filter_non_backbone_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Keep non-local secondary-structure contacts; sequence continuity is represented separately."""
    if edge_index.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    filtered = [
        (int(src), int(dst))
        for src, dst in edge_index.T.tolist()
        if int(src) != int(dst) and abs(int(src) - int(dst)) > 1
    ]
    if not filtered:
        return _construct_primary_graph(num_nodes, window_size=1)
    return _deduplicate_edges(torch.tensor(filtered, dtype=torch.long).T.contiguous(), num_nodes)


def _resolve_view_edges(
    secondary_edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    secondary_contacts = _filter_non_backbone_edges(secondary_edge_index, num_nodes)
    return (
        _construct_primary_graph(num_nodes, window_size=13),
        secondary_edge_index,
        secondary_contacts if secondary_contacts.numel() > 0 else secondary_edge_index,
    )


def _graph_count(slices: dict[str, torch.Tensor]) -> int:
    return int(slices["x"].numel() - 1)


def _sequence_offset_for_dataset(dataset_name: str) -> int:
    return 5 if dataset_name in {"train60", "test18"} else 0


def _extract_sequence(main_x: torch.Tensor, dataset_name: str) -> str:
    offset = _sequence_offset_for_dataset(dataset_name)
    residue_ids = main_x[:, offset : offset + 4].argmax(dim=-1).tolist()
    alphabet = "AUCG"
    return "".join(alphabet[idx] for idx in residue_ids)


def _load_ligand_candidates(metadata_csv_path: str) -> tuple[dict[str, int], dict[str, Counter[str]]]:
    ligand_codes = {"[UNK]"}
    sequence_to_ligands: dict[str, Counter[str]] = defaultdict(Counter)
    with open(metadata_csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ligand_set = ast.literal_eval(row["Ligand"])
            if not ligand_set:
                continue
            sequence = row["Sequence"]
            for ligand_code in ligand_set:
                ligand_codes.add(ligand_code)
                sequence_to_ligands[sequence][ligand_code] += 1
    return {code: idx for idx, code in enumerate(sorted(ligand_codes))}, sequence_to_ligands


def _load_ligand_feature_table(
    ligand_codes: list[str],
    descriptor_cache_path: str,
) -> torch.Tensor:
    cached = torch.load(descriptor_cache_path, map_location="cpu", weights_only=False)
    if cached.get("codes") != ligand_codes:
        raise ValueError("The packaged ligand descriptor cache does not match the benchmark metadata.")
    features = cached.get("features")
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("The packaged ligand descriptor cache is malformed.")
    return features.float()


def _encode_ligand_candidates(
    sequence: str,
    ligand_to_index: dict[str, int],
    sequence_to_ligands: dict[str, Counter[str]],
    ligand_feature_table: torch.Tensor,
    max_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ligand_counter = sequence_to_ligands.get(sequence, Counter())
    if ligand_counter:
        ranked_ligands = ligand_counter.most_common(max_candidates)
        ligand_ids = [ligand_to_index[ligand] for ligand, _ in ranked_ligands]
    else:
        ligand_ids = [ligand_to_index["[UNK]"]]

    padded_ids = torch.zeros(max_candidates, dtype=torch.long)
    padded_mask = torch.zeros(max_candidates, dtype=torch.float32)
    padded_geometry = torch.zeros(max_candidates, ligand_feature_table.size(1), dtype=torch.float32)
    length = min(len(ligand_ids), max_candidates)
    padded_ids[:length] = torch.tensor(ligand_ids[:length], dtype=torch.long)
    padded_mask[:length] = 1.0
    padded_geometry[:length] = ligand_feature_table[padded_ids[:length]]
    return (
        padded_mask.unsqueeze(0),
        padded_geometry.unsqueeze(0),
    )


def load_lqrbind_processed_dataset(
    processed_pt_path: str,
    dataset_name: str,
    ligand_to_index: dict[str, int],
    sequence_to_ligands: dict[str, Counter[str]],
    ligand_feature_table: torch.Tensor,
    max_ligand_candidates: int = 8,
) -> list[Data]:
    packed, slices = torch.load(processed_pt_path, map_location="cpu", weights_only=False)
    dataset = []
    for graph_idx in range(_graph_count(slices)):
        x_start = int(slices["x"][graph_idx].item())
        x_end = int(slices["x"][graph_idx + 1].item())
        y_start = int(slices["y"][graph_idx].item())
        y_end = int(slices["y"][graph_idx + 1].item())
        sub_graph_idx = int(slices["sub_graph"][graph_idx].item())

        main_x = packed.x[x_start:x_end].clone().float()
        labels = packed.y[y_start:y_end].clone().float()
        secondary_graph = packed.sub_graph[sub_graph_idx]
        edge_index_seq, edge_index_sec, edge_index_long = _resolve_view_edges(
            secondary_graph.edge.clone().long(), int(main_x.size(0))
        )
        sequence = _extract_sequence(main_x, dataset_name)
        ligand_mask, ligand_geometry = _encode_ligand_candidates(
            sequence,
            ligand_to_index,
            sequence_to_ligands,
            ligand_feature_table,
            max_ligand_candidates,
        )
        graph = Data(
            x=main_x,
            y=labels,
            edge_index_seq=edge_index_seq,
            edge_index_sec=edge_index_sec,
            edge_index_long=edge_index_long,
            ligand_mask=ligand_mask,
            ligand_geometry=ligand_geometry,
            num_nodes=int(main_x.size(0)),
        )
        dataset.append(graph)
    return dataset


@dataclass
class LQRBindDataBundle:
    train: list[Data]
    validation: list[Data]
    test18: list[Data]
    apo: list[Data]
    conformational: list[Data]
    ligand_vocab_size: int
    ligand_feature_dim: int


def build_lqrbind_data_bundle(
    train60_processed_pt: str,
    test18_processed_pt: str,
    apo_processed_pt: str,
    conformational_processed_pts: list[str],
    metadata_csv_path: str,
    max_ligand_candidates: int = 8,
    seed: int = 42,
) -> LQRBindDataBundle:
    ligand_to_index, sequence_to_ligands = _load_ligand_candidates(metadata_csv_path)
    ordered_codes = sorted(ligand_to_index)
    ligand_feature_table = _load_ligand_feature_table(
        ordered_codes,
        os.path.join(os.path.dirname(metadata_csv_path), "ligand_descriptors.pt"),
    )

    load_kwargs = dict(
        ligand_to_index=ligand_to_index,
        sequence_to_ligands=sequence_to_ligands,
        ligand_feature_table=ligand_feature_table,
        max_ligand_candidates=max_ligand_candidates,
    )
    train60 = load_lqrbind_processed_dataset(train60_processed_pt, "train60", **load_kwargs)
    train_indices, validation_indices = train_test_split(
        list(range(len(train60))), test_size=0.1, random_state=seed
    )
    train = [train60[idx] for idx in train_indices]
    validation = [train60[idx] for idx in validation_indices]
    test18 = load_lqrbind_processed_dataset(test18_processed_pt, "test18", **load_kwargs)
    apo = load_lqrbind_processed_dataset(apo_processed_pt, "apo", **load_kwargs)
    conformational = []
    for split_idx, processed_pt in enumerate(conformational_processed_pts):
        conformational.extend(
            load_lqrbind_processed_dataset(processed_pt, f"conformational_{split_idx}", **load_kwargs)
        )
    return LQRBindDataBundle(
        train=train,
        validation=validation,
        test18=test18,
        apo=apo,
        conformational=conformational,
        ligand_vocab_size=len(ligand_to_index),
        ligand_feature_dim=int(ligand_feature_table.size(1)),
    )
