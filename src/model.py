from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax as graph_softmax

def _complex_normalize(states: torch.Tensor) -> torch.Tensor:
    return states / states.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _edge_mean(node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return node_features

    src, dst = edge_index
    aggregated = node_features.new_zeros(node_features.size(0), node_features.size(1))
    counts = node_features.new_zeros(node_features.size(0), 1)
    aggregated.index_add_(0, dst, node_features[src])
    counts.index_add_(0, dst, torch.ones_like(dst, dtype=node_features.dtype).unsqueeze(-1))
    neighborhood_mean = aggregated / counts.clamp_min(1.0)
    return torch.where(counts > 0, neighborhood_mean, node_features)


def _edge_attention_entropy(query: torch.Tensor, key: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return query.new_zeros(query.size(0), 1)

    src, dst = edge_index
    logits = (query[dst] * key[src]).sum(dim=-1) / math.sqrt(float(query.size(-1)))
    weights = graph_softmax(logits, dst)
    entropy_terms = -(weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()).unsqueeze(-1)
    entropy = query.new_zeros(query.size(0), 1)
    counts = query.new_zeros(query.size(0), 1)
    entropy.index_add_(0, dst, entropy_terms)
    counts.index_add_(0, dst, torch.ones_like(entropy_terms))
    normalizer = torch.log(counts.clamp_min(2.0))
    return entropy / normalizer.clamp_min(1.0)


def _topk_candidate_mask(scores: torch.Tensor, batch_index: torch.Tensor, k: int) -> torch.Tensor:
    mask = scores.new_zeros(scores.size(0), 1)
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
    for graph_id in range(num_graphs):
        node_ids = torch.nonzero(batch_index == graph_id, as_tuple=False).squeeze(-1)
        if node_ids.numel() == 0:
            continue
        graph_scores = scores[node_ids]
        topk = min(k, graph_scores.numel())
        selected = torch.topk(graph_scores, k=topk, dim=0).indices
        mask[node_ids[selected], 0] = 1.0
    return mask


class _CheckpointEdgeParameters(nn.Module):
    """Checkpoint-compatible parameter container retained by the released model."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 7),
            nn.Linear(hidden_dim * 2 + 7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )


class QuantumStructuralEncodingLite(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int, dropout: float):
        super().__init__()
        self.state_dim = state_dim
        self.amplitude_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, state_dim))
        self.phase_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, state_dim))
        self.context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim + 7),
            nn.Linear(hidden_dim + 7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def _encode_complex(self, features: torch.Tensor) -> torch.Tensor:
        amplitudes = F.softplus(self.amplitude_proj(features)) + 1e-4
        phases = math.pi * torch.tanh(self.phase_proj(features))
        states = amplitudes.to(torch.complex64) * torch.exp(1j * phases.to(torch.complex64))
        return _complex_normalize(states)

    def forward(
        self,
        node_context: torch.Tensor,
        edge_index_sec: torch.Tensor,
        edge_index_long: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_context = _edge_mean(node_context, edge_index_sec)
        inter_chain_context = _edge_mean(node_context, edge_index_long)
        multi_hop_context = _edge_mean(spatial_context, edge_index_sec)

        node_state = self._encode_complex(node_context)
        spatial_state = self._encode_complex(spatial_context)
        inter_state = self._encode_complex(inter_chain_context)
        multi_hop_state = self._encode_complex(multi_hop_context)

        spatial_overlap = (node_state * spatial_state.conj()).sum(dim=-1, keepdim=True)
        inter_overlap = (node_state * inter_state.conj()).sum(dim=-1, keepdim=True)
        multi_hop_overlap = (node_state * multi_hop_state.conj()).sum(dim=-1, keepdim=True)
        branch_overlap = (spatial_state * inter_state.conj()).sum(dim=-1, keepdim=True)

        spatial_return = spatial_overlap.abs().pow(2)
        inter_return = inter_overlap.abs().pow(2)
        multi_hop_return = multi_hop_overlap.abs().pow(2)
        branch_coherence = branch_overlap.abs()
        phase_agreement = 0.5 * (1.0 + torch.cos(torch.angle(spatial_overlap) - torch.angle(inter_overlap)))

        spatial_entropy = _edge_attention_entropy(node_context, spatial_context, edge_index_sec)
        inter_entropy = _edge_attention_entropy(node_context, inter_chain_context, edge_index_long)

        structural_stats = torch.cat(
            [
                spatial_return,
                inter_return,
                multi_hop_return,
                branch_coherence,
                phase_agreement,
                spatial_entropy,
                inter_entropy,
            ],
            dim=-1,
        )
        structural_context = self.context_proj(
            torch.cat([node_context, spatial_context, inter_chain_context, multi_hop_context], dim=-1)
        )
        structural_hidden = self.output_proj(torch.cat([structural_context, structural_stats], dim=-1))
        return structural_hidden, structural_stats


class QuantumProductKernelStructuralEncodingLite(nn.Module):
    """Tensor-product Hilbert embedding used by the QSE structural encoder."""

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        dropout: float,
        num_factors: int = 4,
        factor_dim: int | None = None,
    ):
        super().__init__()
        self.num_factors = max(2, int(num_factors))
        if factor_dim is None:
            factor_dim = max(4, min(8, int(state_dim)))
        self.factor_dim = max(2, int(factor_dim))
        self.factor_size = self.num_factors * self.factor_dim
        self.amplitude_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.factor_size))
        self.phase_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.factor_size))
        self.factor_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_factors),
            nn.Sigmoid(),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
        )
        self.product_proj = nn.Sequential(
            nn.LayerNorm(self.factor_size * 6),
            nn.Linear(self.factor_size * 6, hidden_dim),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 7),
            nn.Linear(hidden_dim * 2 + 7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def _factor_state(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amplitudes = F.softplus(self.amplitude_proj(features)).view(-1, self.num_factors, self.factor_dim) + 1e-4
        phases = math.pi * torch.tanh(self.phase_proj(features)).view(-1, self.num_factors, self.factor_dim)
        real = amplitudes * torch.cos(phases)
        imag = amplitudes * torch.sin(phases)
        norm = torch.sqrt((real.square() + imag.square()).sum(dim=-1, keepdim=True).clamp_min(1e-6))
        return real / norm, imag / norm

    def _factor_fidelity(
        self,
        real_a: torch.Tensor,
        imag_a: torch.Tensor,
        real_b: torch.Tensor,
        imag_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_overlap = (real_a * real_b + imag_a * imag_b).sum(dim=-1)
        imag_overlap = (imag_a * real_b - real_a * imag_b).sum(dim=-1)
        factor_fidelity = (real_overlap.square() + imag_overlap.square()).clamp_min(1e-6)
        kernel = torch.exp(factor_fidelity.log().mean(dim=-1, keepdim=True))
        phase = torch.atan2(imag_overlap, real_overlap)
        return kernel, phase, factor_fidelity

    def forward(
        self,
        node_context: torch.Tensor,
        edge_index_sec: torch.Tensor,
        edge_index_long: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_context = _edge_mean(node_context, edge_index_sec)
        inter_chain_context = _edge_mean(node_context, edge_index_long)
        multi_hop_context = _edge_mean(spatial_context, edge_index_sec)
        combined_context = torch.cat([node_context, spatial_context, inter_chain_context, multi_hop_context], dim=-1)

        node_real, node_imag = self._factor_state(node_context)
        spatial_real, spatial_imag = self._factor_state(spatial_context)
        inter_real, inter_imag = self._factor_state(inter_chain_context)
        multi_hop_real, multi_hop_imag = self._factor_state(multi_hop_context)

        spatial_kernel, spatial_phase, spatial_factors = self._factor_fidelity(
            node_real,
            node_imag,
            spatial_real,
            spatial_imag,
        )
        inter_kernel, inter_phase, inter_factors = self._factor_fidelity(
            node_real,
            node_imag,
            inter_real,
            inter_imag,
        )
        multi_hop_kernel, _, multi_hop_factors = self._factor_fidelity(
            node_real,
            node_imag,
            multi_hop_real,
            multi_hop_imag,
        )
        branch_kernel, branch_phase, branch_factors = self._factor_fidelity(
            spatial_real,
            spatial_imag,
            inter_real,
            inter_imag,
        )
        factor_gate = self.factor_gate(combined_context)
        gated_kernel = (
            factor_gate * (spatial_factors + inter_factors + multi_hop_factors + branch_factors) * 0.25
        ).sum(dim=-1, keepdim=True) / factor_gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        phase_agreement = 0.5 * (1.0 + torch.cos(spatial_phase - inter_phase).mean(dim=-1, keepdim=True))
        spatial_entropy = _edge_attention_entropy(node_context, spatial_context, edge_index_sec)
        inter_entropy = _edge_attention_entropy(node_context, inter_chain_context, edge_index_long)
        structural_stats = torch.cat(
            [
                0.5 * (spatial_kernel + gated_kernel),
                0.5 * (inter_kernel + gated_kernel),
                multi_hop_kernel,
                branch_kernel,
                phase_agreement,
                spatial_entropy,
                inter_entropy,
            ],
            dim=-1,
        )
        structural_context = self.context_proj(combined_context)
        product_features = torch.cat(
            [
                node_real.flatten(start_dim=1),
                node_imag.flatten(start_dim=1),
                spatial_real.flatten(start_dim=1),
                spatial_imag.flatten(start_dim=1),
                inter_real.flatten(start_dim=1),
                inter_imag.flatten(start_dim=1),
            ],
            dim=-1,
        )
        product_context = self.product_proj(product_features)
        structural_hidden = self.output_proj(torch.cat([structural_context, product_context, structural_stats], dim=-1))
        return structural_hidden, structural_stats


class QuantumDensityKernelStructuralEncodingLite(nn.Module):
    """Mixed-state quantum kernel structural encoder with the QSE interface."""

    def __init__(self, hidden_dim: int, state_dim: int, dropout: float):
        super().__init__()
        self.state_dim = max(4, int(state_dim))
        self.num_components = 4
        self.real_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.state_dim))
        self.imag_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.state_dim))
        self.component_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * self.num_components),
            nn.Linear(hidden_dim * self.num_components, hidden_dim),
            nn.GELU(),
        )
        self.density_proj = nn.Sequential(
            nn.LayerNorm(self.state_dim * 3 + self.num_components),
            nn.Linear(self.state_dim * 3 + self.num_components, hidden_dim),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 7),
            nn.Linear(hidden_dim * 2 + 7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def _complex_parts(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real = self.real_proj(features)
        imag = self.imag_proj(features)
        norm = torch.sqrt((real.square() + imag.square()).sum(dim=-1, keepdim=True).clamp_min(1e-6))
        return real / norm, imag / norm

    def _fidelity(
        self,
        real_a: torch.Tensor,
        imag_a: torch.Tensor,
        real_b: torch.Tensor,
        imag_b: torch.Tensor,
    ) -> torch.Tensor:
        real_overlap = (real_a * real_b + imag_a * imag_b).sum(dim=-1, keepdim=True)
        imag_overlap = (imag_a * real_b - real_a * imag_b).sum(dim=-1, keepdim=True)
        return real_overlap.square() + imag_overlap.square()

    def _density_expectation(
        self,
        query_real: torch.Tensor,
        query_imag: torch.Tensor,
        component_real: torch.Tensor,
        component_imag: torch.Tensor,
        component_weights: torch.Tensor,
    ) -> torch.Tensor:
        fidelity = self._fidelity(
            query_real.unsqueeze(1),
            query_imag.unsqueeze(1),
            component_real,
            component_imag,
        ).squeeze(-1)
        return (component_weights * fidelity).sum(dim=-1, keepdim=True)

    def forward(
        self,
        node_context: torch.Tensor,
        edge_index_sec: torch.Tensor,
        edge_index_long: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_context = _edge_mean(node_context, edge_index_sec)
        inter_chain_context = _edge_mean(node_context, edge_index_long)
        multi_hop_context = _edge_mean(spatial_context, edge_index_sec)
        components = torch.stack([node_context, spatial_context, inter_chain_context, multi_hop_context], dim=1)
        component_logits = self.component_score(components).squeeze(-1)
        component_weights = torch.softmax(component_logits, dim=-1)

        real_parts = []
        imag_parts = []
        for index in range(self.num_components):
            real, imag = self._complex_parts(components[:, index])
            real_parts.append(real)
            imag_parts.append(imag)
        component_real = torch.stack(real_parts, dim=1)
        component_imag = torch.stack(imag_parts, dim=1)
        node_real, node_imag = real_parts[0], imag_parts[0]
        spatial_real, spatial_imag = real_parts[1], imag_parts[1]
        inter_real, inter_imag = real_parts[2], imag_parts[2]
        multi_real, multi_imag = real_parts[3], imag_parts[3]

        node_return = self._density_expectation(node_real, node_imag, component_real, component_imag, component_weights)
        spatial_return = self._density_expectation(
            spatial_real,
            spatial_imag,
            component_real,
            component_imag,
            component_weights,
        )
        inter_return = self._density_expectation(inter_real, inter_imag, component_real, component_imag, component_weights)
        multi_return = self._density_expectation(multi_real, multi_imag, component_real, component_imag, component_weights)
        pairwise_fidelity = self._fidelity(
            component_real.unsqueeze(2),
            component_imag.unsqueeze(2),
            component_real.unsqueeze(1),
            component_imag.unsqueeze(1),
        ).squeeze(-1)
        purity = (
            component_weights.unsqueeze(2) * component_weights.unsqueeze(1) * pairwise_fidelity
        ).sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
        spatial_entropy = _edge_attention_entropy(node_context, spatial_context, edge_index_sec)
        inter_entropy = _edge_attention_entropy(node_context, inter_chain_context, edge_index_long)
        structural_stats = torch.cat(
            [
                node_return,
                spatial_return,
                inter_return,
                0.5 * (multi_return + purity),
                purity,
                spatial_entropy,
                inter_entropy,
            ],
            dim=-1,
        )

        density_prob = (
            component_weights.unsqueeze(-1) * (component_real.square() + component_imag.square())
        ).sum(dim=1)
        mean_real = (component_weights.unsqueeze(-1) * component_real).sum(dim=1)
        mean_imag = (component_weights.unsqueeze(-1) * component_imag).sum(dim=1)
        density_context = self.density_proj(torch.cat([density_prob, mean_real, mean_imag, component_weights], dim=-1))
        structural_context = self.context_proj(components.flatten(start_dim=1))
        structural_hidden = self.output_proj(torch.cat([structural_context, density_context, structural_stats], dim=-1))
        return structural_hidden, structural_stats


class QuantumAdaptiveMultiKernelStructuralEncodingLite(nn.Module):
    """Adaptive mixture of lightweight quantum structural kernels."""

    def __init__(self, hidden_dim: int, state_dim: int, dropout: float):
        super().__init__()
        self.single_encoder = QuantumStructuralEncodingLite(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            dropout=dropout,
        )
        self.product_encoder = QuantumProductKernelStructuralEncodingLite(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            dropout=dropout,
        )
        self.density_encoder = QuantumDensityKernelStructuralEncodingLite(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            dropout=dropout,
        )
        self.kernel_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + 21),
            nn.Linear(hidden_dim * 4 + 21, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.refine_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim + 7),
            nn.Linear(hidden_dim + 7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        final_gate = self.kernel_gate[-1]
        nn.init.zeros_(final_gate.weight)
        nn.init.zeros_(final_gate.bias)

    def forward(
        self,
        node_context: torch.Tensor,
        edge_index_sec: torch.Tensor,
        edge_index_long: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_hidden, single_stats = self.single_encoder(
            node_context=node_context,
            edge_index_sec=edge_index_sec,
            edge_index_long=edge_index_long,
        )
        product_hidden, product_stats = self.product_encoder(
            node_context=node_context,
            edge_index_sec=edge_index_sec,
            edge_index_long=edge_index_long,
        )
        density_hidden, density_stats = self.density_encoder(
            node_context=node_context,
            edge_index_sec=edge_index_sec,
            edge_index_long=edge_index_long,
        )
        gate_input = torch.cat(
            [
                node_context,
                single_hidden,
                product_hidden,
                density_hidden,
                single_stats,
                product_stats,
                density_stats,
            ],
            dim=-1,
        )
        kernel_weights = torch.softmax(self.kernel_gate(gate_input), dim=-1)
        hidden_stack = torch.stack([single_hidden, product_hidden, density_hidden], dim=1)
        stats_stack = torch.stack([single_stats, product_stats, density_stats], dim=1)
        structural_hidden = (kernel_weights.unsqueeze(-1) * hidden_stack).sum(dim=1)
        structural_stats = (kernel_weights.unsqueeze(-1) * stats_stack).sum(dim=1)
        structural_hidden = self.refine_proj(torch.cat([structural_hidden, structural_stats], dim=-1))
        return structural_hidden, structural_stats


class LQRBindQSEReadout(nn.Module):
    """Ligand-conditioned QSE residue readout used by LQRBind."""

    def __init__(
        self,
        hidden_dim: int,
        num_scales: int,
        state_dim: int,
        ligand_vocab_size: int,
        ligand_feature_dim: int,
        dropout: float,
        candidate_topk: int = 18,
        residual_strength: float = 0.32,
        gate_bias: float = -1.70,
        fidelity_gate_scale: float = 0.16,
        residual_temperature: float = 1.0,
        center_blend: float = 0.45,
        uncertainty_scale: float = 6.0,
        score_calibration_blend: float = 0.18,
        mean_shift_blend: float = 0.35,
        target_candidate_logit: float = 0.0,
        target_candidate_std: float = 1.0,
        min_temperature: float = 0.75,
        max_temperature: float = 1.35,
        std_gate_scale: float = 4.0,
        mean_gate_scale: float = 4.0,
        mean_shift_threshold: float = 0.35,
        min_graph_gate: float = 0.02,
        rate_shift_blend: float = 0.25,
        rate_gate_scale: float = 4.0,
        rate_shift_threshold: float = 0.15,
        max_rate_shift: float = 1.0,
        min_candidate_rate: float = 0.30,
        max_candidate_rate: float = 0.44,
        uncertainty_rate_scale: float = 5.0,
        uncertainty_rate_center: float = 0.55,
        compression_rate_scale: float = 4.0,
        phase_confidence_scale: float = 8.0,
        phase_confidence_center: float = 0.10,
        coherent_rate_blend: float = 0.65,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.candidate_topk = candidate_topk
        self.structural_state_dim = min(state_dim, 8)
        self.ligand_state_dim = max(4, min(state_dim, hidden_dim // 2))
        self.scale_embeddings = nn.Parameter(torch.empty(num_scales, hidden_dim))
        self.node_context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.scale_score = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        with torch.random.fork_rng(devices=[]):
            self.structure_encoder = QuantumAdaptiveMultiKernelStructuralEncodingLite(
                hidden_dim=hidden_dim,
                state_dim=self.structural_state_dim,
                dropout=dropout,
            )
        self.base_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + 7),
            nn.Linear(hidden_dim * 4 + 7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Inactive parameter slots retained for strict checkpoint compatibility.
        self.ligand_embedding = nn.Embedding(ligand_vocab_size, hidden_dim)
        self.ligand_geometry_proj = nn.Sequential(
            nn.LayerNorm(ligand_feature_dim),
            nn.Linear(ligand_feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.ligand_token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.ligand_context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.ligand_score = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.rerank_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + 11),
            nn.Linear(hidden_dim * 4 + 11, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.rerank_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + 11),
            nn.Linear(hidden_dim * 4 + 11, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.reset_parameters()
        self.max_residual_strength = residual_strength
        self._reset_adapter(gate_bias)
        self.quantum_state_dim = self.ligand_state_dim
        self.fidelity_gate_scale = float(fidelity_gate_scale)
        phase = torch.linspace(0.0, math.pi, self.quantum_state_dim)
        self.register_buffer("phase_pattern", phase)
        self.phase_logits = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
        self.residual_temperature = float(residual_temperature)
        self.center_blend = float(center_blend)
        self.uncertainty_scale = float(uncertainty_scale)
        self.score_calibration_blend = float(score_calibration_blend)
        self.mean_shift_blend = float(mean_shift_blend)
        self.target_candidate_logit = float(target_candidate_logit)
        self.target_candidate_std = float(target_candidate_std)
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.std_gate_scale = float(std_gate_scale)
        self.mean_gate_scale = float(mean_gate_scale)
        self.mean_shift_threshold = float(mean_shift_threshold)
        self.min_graph_gate = float(min_graph_gate)
        self.rate_shift_blend = float(rate_shift_blend)
        self.rate_gate_scale = float(rate_gate_scale)
        self.rate_shift_threshold = float(rate_shift_threshold)
        self.max_rate_shift = float(max_rate_shift)
        self.min_candidate_rate = float(min_candidate_rate)
        self.max_candidate_rate = float(max_candidate_rate)
        self.uncertainty_rate_scale = float(uncertainty_rate_scale)
        self.uncertainty_rate_center = float(uncertainty_rate_center)
        self.compression_rate_scale = float(compression_rate_scale)
        self.phase_confidence_scale = float(phase_confidence_scale)
        self.phase_confidence_center = float(phase_confidence_center)
        self.coherent_rate_blend = float(coherent_rate_blend)

    def reset_parameters(self):
        nn.init.normal_(self.scale_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.ligand_embedding.weight, mean=0.0, std=0.02)

    def _reset_adapter(self, gate_bias: float):
        final_delta = self.rerank_head[-1]
        nn.init.zeros_(final_delta.weight)
        nn.init.zeros_(final_delta.bias)
        gate_linear = self.rerank_gate[-2]
        nn.init.zeros_(gate_linear.weight)
        nn.init.constant_(gate_linear.bias, gate_bias)

    def _complex_parts_from_hidden(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = source[:, : 2 * self.quantum_state_dim]
        real, imag = state.chunk(2, dim=-1)
        norm = torch.sqrt((real.square() + imag.square()).sum(dim=-1, keepdim=True).clamp_min(1e-6))
        return real / norm, imag / norm

    def _phase_shifted_ligand_states(
        self,
        ligand_real: torch.Tensor,
        ligand_imag: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        phase = self.phase_pattern.unsqueeze(0).to(ligand_real.device)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        cos_neg = torch.cos(-phase)
        sin_neg = torch.sin(-phase)
        positive_real = ligand_real * cos_phase - ligand_imag * sin_phase
        positive_imag = ligand_real * sin_phase + ligand_imag * cos_phase
        negative_real = ligand_real * cos_neg - ligand_imag * sin_neg
        negative_imag = ligand_real * sin_neg + ligand_imag * cos_neg
        return [
            (ligand_real, ligand_imag),
            (positive_real, positive_imag),
            (negative_real, negative_imag),
        ]

    def _relative_fidelity_gate(
        self,
        node_context: torch.Tensor,
        structural_hidden: torch.Tensor,
        scale_summary: torch.Tensor,
        ligand_context: torch.Tensor,
        ligand_geometry_context: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        rna_real, rna_imag = self._complex_parts_from_hidden(
            node_context + 0.5 * structural_hidden + 0.25 * scale_summary
        )
        ligand_real, ligand_imag = self._complex_parts_from_hidden(ligand_context + ligand_geometry_context)
        fidelity_terms = []
        for shifted_real, shifted_imag in self._phase_shifted_ligand_states(ligand_real, ligand_imag):
            real_overlap = (rna_real * shifted_real + rna_imag * shifted_imag).sum(dim=-1, keepdim=True)
            imag_overlap = (rna_imag * shifted_real - rna_real * shifted_imag).sum(dim=-1, keepdim=True)
            fidelity_terms.append(real_overlap.square() + imag_overlap.square())
        fidelity_stack = torch.stack(fidelity_terms, dim=-1)
        phase_weights = torch.softmax(self.phase_logits, dim=0).view(1, 1, -1)
        fidelity = (fidelity_stack * phase_weights).sum(dim=-1)
        graph_fidelity = global_mean_pool(fidelity, batch_index)[batch_index]
        relative_fidelity = fidelity - graph_fidelity
        return 1.0 + self.fidelity_gate_scale * torch.tanh(4.0 * relative_fidelity).squeeze(-1)

    def _center_candidate_residual(
        self,
        raw_residual: torch.Tensor,
        candidate_mask: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        mask = candidate_mask.squeeze(-1)
        candidate_sum = global_add_pool((raw_residual * mask).unsqueeze(-1), batch_index)
        candidate_count = global_add_pool(mask.unsqueeze(-1), batch_index).clamp_min(1.0)
        candidate_mean = candidate_sum / candidate_count
        centered = (raw_residual - candidate_mean[batch_index].squeeze(-1)) * mask
        return centered / max(self.residual_temperature, 1e-6)

    def forward(self, states: list[torch.Tensor], batch) -> torch.Tensor:
        batch_index = batch.batch
        tokens = torch.stack(states, dim=1) + self.scale_embeddings.unsqueeze(0)
        node_context = self.node_context_proj(torch.cat([states[0], states[-1]], dim=-1))
        graph_context = global_mean_pool(node_context, batch_index)
        structural_hidden, structural_stats = self.structure_encoder(
            node_context=node_context,
            edge_index_sec=batch.edge_index_sec,
            edge_index_long=batch.edge_index_long,
        )

        graph_query = graph_context[batch_index].unsqueeze(1).expand(-1, self.num_scales, -1)
        scale_logits = self.scale_score(torch.cat([tokens, graph_query], dim=-1)).squeeze(-1)
        scale_weights = torch.softmax(scale_logits, dim=1).unsqueeze(-1)
        scale_summary = (scale_weights * tokens).sum(dim=1)
        scale_max = tokens.max(dim=1).values

        base_features = torch.cat(
            [node_context, structural_hidden, scale_summary, scale_max, structural_stats],
            dim=-1,
        )
        base_logit = self.base_head(base_features).squeeze(-1)

        ligand_mask = batch.ligand_mask.float()
        ligand_geometry = batch.ligand_geometry.float()
        ligand_geometry_hidden = self.ligand_geometry_proj(ligand_geometry)

        ligand_weights = ligand_mask / ligand_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        ligand_geometry_graph = (ligand_weights.unsqueeze(-1) * ligand_geometry_hidden).sum(dim=1)
        valid_candidates = ligand_mask.sum(dim=-1, keepdim=True)
        entropy_norm = torch.where(valid_candidates > 1.0, valid_candidates.log(), torch.ones_like(valid_candidates))
        ligand_entropy_graph = -(
            ligand_weights.clamp_min(1e-6) * ligand_weights.clamp_min(1e-6).log()
        ).sum(dim=-1, keepdim=True) / entropy_norm

        ligand_geometry_context = ligand_geometry_graph[batch_index]
        ligand_context = ligand_geometry_context
        ligand_entropy = ligand_entropy_graph[batch_index]
        ligand_known = torch.ones_like(ligand_entropy)
        geometry_alignment = torch.sigmoid(
            (node_context * ligand_geometry_context).sum(dim=-1, keepdim=True)
            / math.sqrt(float(node_context.size(-1)))
        )
        quantum_gate = self._relative_fidelity_gate(
            node_context=node_context,
            structural_hidden=structural_hidden,
            scale_summary=scale_summary,
            ligand_context=ligand_context,
            ligand_geometry_context=ligand_geometry_context,
            batch_index=batch_index,
        )
        candidate_uncertainty = torch.exp(-base_logit.abs()).unsqueeze(-1)
        candidate_mask = _topk_candidate_mask(base_logit, batch_index, self.candidate_topk)

        rerank_features = torch.cat(
            [
                node_context,
                structural_hidden,
                ligand_context,
                ligand_geometry_context,
                structural_stats,
                ligand_entropy,
                ligand_known,
                geometry_alignment,
                candidate_uncertainty,
            ],
            dim=-1,
        )
        rerank_delta = self.rerank_head(rerank_features).squeeze(-1)
        rerank_gate = self.rerank_gate(rerank_features).squeeze(-1)
        raw_residual = rerank_gate * quantum_gate * rerank_delta
        centered_residual = self._center_candidate_residual(raw_residual, candidate_mask, batch_index)
        calibrated_residual = self._calibrated_candidate_residual(
            raw_residual=raw_residual,
            centered_residual=centered_residual,
            candidate_uncertainty=candidate_uncertainty,
            candidate_mask=candidate_mask,
            batch_index=batch_index,
        )
        final_logit = base_logit + self.max_residual_strength * calibrated_residual
        return self._calibrate_final_logit(
            final_logit=final_logit,
            base_logit=base_logit,
            candidate_uncertainty=candidate_uncertainty,
            candidate_mask=candidate_mask,
            batch_index=batch_index,
        )

    def _calibrated_candidate_residual(
        self,
        raw_residual: torch.Tensor,
        centered_residual: torch.Tensor,
        candidate_uncertainty: torch.Tensor,
        candidate_mask: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        mask = candidate_mask.squeeze(-1)
        uncertainty = candidate_uncertainty.squeeze(-1)
        candidate_count = global_add_pool(mask.unsqueeze(-1), batch_index).clamp_min(1.0)
        uncertainty_mean = global_add_pool((uncertainty * mask).unsqueeze(-1), batch_index) / candidate_count
        uncertainty_gate = torch.sigmoid(
            self.uncertainty_scale * (uncertainty - uncertainty_mean[batch_index].squeeze(-1))
        )
        calibration_gate = (self.center_blend * uncertainty_gate * mask).clamp(0.0, 1.0)
        raw_candidate_residual = raw_residual * mask
        return (1.0 - calibration_gate) * raw_candidate_residual + calibration_gate * centered_residual

    def _calibrate_final_logit(
        self,
        final_logit: torch.Tensor,
        base_logit: torch.Tensor,
        candidate_uncertainty: torch.Tensor,
        candidate_mask: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        mask = candidate_mask.squeeze(-1)
        candidate_count = global_add_pool(mask.unsqueeze(-1), batch_index).clamp_min(1.0)
        candidate_mean = global_add_pool((final_logit * mask).unsqueeze(-1), batch_index) / candidate_count
        centered = final_logit - candidate_mean[batch_index].squeeze(-1)
        candidate_var = global_add_pool((centered.square() * mask).unsqueeze(-1), batch_index) / candidate_count
        candidate_std = candidate_var.sqrt().clamp_min(1e-3)
        temperature = (candidate_std / self.target_candidate_std).clamp(
            self.min_temperature,
            self.max_temperature,
        )
        scaled = candidate_mean[batch_index].squeeze(-1) + centered / temperature[batch_index].squeeze(-1)
        shifted = scaled + self.mean_shift_blend * (
            self.target_candidate_logit - candidate_mean[batch_index].squeeze(-1)
        )

        std_gate = torch.sigmoid(
            self.std_gate_scale * (self.target_candidate_std - candidate_std).squeeze(-1)
        )
        mean_deviation = (candidate_mean - self.target_candidate_logit).abs().squeeze(-1)
        mean_gate = torch.sigmoid(self.mean_gate_scale * (mean_deviation - self.mean_shift_threshold))
        graph_gate = torch.maximum(std_gate, mean_gate).clamp(self.min_graph_gate, 1.0)
        node_blend = self.score_calibration_blend * graph_gate[batch_index] * mask
        candidate_calibrated = (1.0 - node_blend) * final_logit + node_blend * shifted

        candidate_prob = torch.sigmoid(candidate_calibrated)
        mean_prob = global_add_pool((candidate_prob * mask).unsqueeze(-1), batch_index) / candidate_count
        mean_prob = mean_prob.clamp(1e-4, 1.0 - 1e-4)

        uncertainty = candidate_uncertainty.squeeze(-1)
        uncertainty_mean = global_add_pool((uncertainty * mask).unsqueeze(-1), batch_index) / candidate_count
        uncertainty_gate = torch.sigmoid(
            self.uncertainty_rate_scale * (uncertainty_mean - self.uncertainty_rate_center)
        )
        compression_gate = torch.sigmoid(
            self.compression_rate_scale * (self.target_candidate_std - candidate_std)
        )
        adaptive_gate = torch.maximum(uncertainty_gate, compression_gate)
        coherent_gate = torch.sqrt((uncertainty_gate * compression_gate).clamp_min(1e-6))

        qse_residual = final_logit - base_logit
        residual_mean = global_add_pool((qse_residual * mask).unsqueeze(-1), batch_index) / candidate_count
        residual_centered = qse_residual - residual_mean[batch_index].squeeze(-1)
        residual_var = global_add_pool((residual_centered.square() * mask).unsqueeze(-1), batch_index) / candidate_count
        residual_std = residual_var.clamp_min(1e-6).sqrt()
        phase_confidence = torch.sigmoid(
            self.phase_confidence_scale * (residual_std - self.phase_confidence_center)
        )

        guarded_gate = (
            (1.0 - self.coherent_rate_blend) * adaptive_gate
            + self.coherent_rate_blend * coherent_gate
        )
        blended_gate = phase_confidence * adaptive_gate + (1.0 - phase_confidence) * guarded_gate
        target_rate = self.min_candidate_rate + (self.max_candidate_rate - self.min_candidate_rate) * blended_gate
        target_rate = target_rate.clamp(1e-4, 1.0 - 1e-4)

        target_logit = torch.log(target_rate / (1.0 - target_rate))
        mean_prob_logit = torch.log(mean_prob / (1.0 - mean_prob))
        rate_shift = (target_logit - mean_prob_logit).clamp(
            -self.max_rate_shift,
            self.max_rate_shift,
        )
        rate_gate = torch.sigmoid(
            self.rate_gate_scale * (rate_shift.abs().squeeze(-1) - self.rate_shift_threshold)
        )
        node_shift = (
            self.rate_shift_blend
            * graph_gate[batch_index]
            * rate_gate[batch_index]
            * rate_shift[batch_index].squeeze(-1)
        )
        candidate_calibrated = candidate_calibrated + node_shift * mask
        return torch.where(mask > 0.0, candidate_calibrated, final_logit)


class TTNEntanglementFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        bond_dim: int,
        dropout: float,
        use_context_gate: bool = True,
    ):
        super().__init__()
        self.bond_dim = bond_dim
        self.use_context_gate = use_context_gate
        self.leaf_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, bond_dim),
                    nn.Tanh(),
                )
                for _ in range(3)
            ]
        )
        self.left_core = nn.Parameter(torch.empty(bond_dim, bond_dim, bond_dim))
        self.right_core = nn.Parameter(torch.empty(bond_dim, bond_dim, bond_dim))
        self.output_proj = nn.Sequential(
            nn.Linear(bond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.bond_dim)
        with torch.no_grad():
            self.left_core.normal_(mean=0.0, std=std)
            self.right_core.normal_(mean=0.0, std=std)

    def forward(
        self,
        seq_features: torch.Tensor,
        sec_features: torch.Tensor,
        long_features: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        seq_leaf, sec_leaf, long_leaf = [
            proj(features)
            for proj, features in zip(
                self.leaf_projs,
                (seq_features, sec_features, long_features),
            )
        ]
        left_state = torch.tanh(
            torch.einsum("bi,bj,ijk->bk", seq_leaf, sec_leaf, self.left_core)
        )
        fused_state = torch.tanh(
            torch.einsum("bi,bj,ijk->bk", left_state, long_leaf, self.right_core)
        )
        fused_state = self.output_proj(fused_state)
        if not self.use_context_gate:
            return fused_state
        gate = self.context_gate(torch.cat([context, fused_state], dim=-1))
        return gate * fused_state + (1.0 - gate) * context


class MultiplexResidueBlock(nn.Module):
    """LQRBind multi-view residue propagation block used by the released model."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        ttn_bond_dim: int,
    ):
        super().__init__()

        # Keep inactive parameter slots from the published training snapshot so
        # the paper checkpoint loads strictly and deterministic retraining keeps
        # the same initialization and serialization order.
        self.seq_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        # Legacy checkpoint names: ``spatial`` stores the secondary-contact
        # branch and ``inter_chain`` stores the long-range-contact branch.
        self.spatial_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.inter_chain_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.agent_conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.spatial_geo_conv = _CheckpointEdgeParameters(hidden_dim, dropout)
        self.inter_geo_conv = _CheckpointEdgeParameters(hidden_dim, dropout)
        self.spatial_frame_conv = _CheckpointEdgeParameters(hidden_dim, dropout)
        self.inter_frame_conv = _CheckpointEdgeParameters(hidden_dim, dropout)
        self.spatial_geo_alpha = nn.Parameter(torch.tensor(-2.2))
        self.inter_geo_alpha = nn.Parameter(torch.tensor(-2.2))
        self.tertiary_reliability_alpha = nn.Parameter(torch.zeros(1))
        self.tertiary_residual_reliability_alpha = nn.Parameter(torch.zeros(1))
        self.spatial_frame_alpha = nn.Parameter(torch.tensor(-2.2))
        self.inter_frame_alpha = nn.Parameter(torch.tensor(-2.2))
        self.agent_evidence_alpha = nn.Parameter(torch.tensor(-2.6))
        self.spatial_role_alpha = nn.Parameter(torch.zeros(1))
        self.inter_role_alpha = nn.Parameter(torch.zeros(1))
        self.seq_branch_norm = nn.LayerNorm(hidden_dim)
        self.spatial_branch_norm = nn.LayerNorm(hidden_dim)
        self.inter_branch_norm = nn.LayerNorm(hidden_dim)
        self.agent_branch_norm = nn.LayerNorm(hidden_dim)
        self.seq_branch_gain = nn.Parameter(torch.ones(1))
        self.spatial_branch_gain = nn.Parameter(torch.ones(1))
        self.inter_branch_gain = nn.Parameter(torch.ones(1))
        self.cross_view_fusion = TTNEntanglementFusion(
            hidden_dim=hidden_dim,
            bond_dim=ttn_bond_dim,
            dropout=dropout,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _apply_conv(conv: GCNConv, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(x)
        return conv(x, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_seq: torch.Tensor,
        edge_index_sec: torch.Tensor,
        edge_index_long: torch.Tensor,
    ) -> torch.Tensor:
        seq_features = self._apply_conv(self.seq_conv, x, edge_index_seq)
        sec_features = self._apply_conv(self.spatial_conv, x, edge_index_sec)
        long_features = self._apply_conv(self.inter_chain_conv, x, edge_index_long)
        entangled_views = self.cross_view_fusion(
            seq_features=seq_features,
            sec_features=sec_features,
            long_features=long_features,
            context=x,
        )
        agent_features = torch.zeros_like(x)
        agent_scale = 0.20 * torch.sigmoid(self.agent_evidence_alpha)
        branch_summary = (
            seq_features + sec_features + long_features + agent_scale * agent_features
        )
        updated = self.mlp(torch.cat([x, entangled_views, branch_summary], dim=-1))
        return self.norm(x + self.dropout(updated))


class LQRBind(nn.Module):
    """Released LQRBind architecture with TTN multi-view propagation and QSE scoring."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        ttn_bond_dim: int = 16,
        quantum_state_dim: int = 16,
        candidate_topk: int = 18,
        ligand_vocab_size: int = 1,
        ligand_feature_dim: int = 0,
        use_seq_view: bool = True,
        use_sec_view: bool = True,
        use_long_view: bool = True,
    ):
        super().__init__()
        self.use_seq_view = use_seq_view
        self.use_sec_view = use_sec_view
        self.use_long_view = use_long_view
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                MultiplexResidueBlock(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    ttn_bond_dim=ttn_bond_dim,
                )
                for _ in range(num_layers)
            ]
        )
        # Parameter-free placeholders retained for byte-identical checkpoint serialization.
        self.post_layer_norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        self.head = LQRBindQSEReadout(
            hidden_dim=hidden_dim,
            num_scales=num_layers + 1,
            state_dim=quantum_state_dim,
            ligand_vocab_size=ligand_vocab_size,
            ligand_feature_dim=ligand_feature_dim,
            dropout=dropout,
            candidate_topk=candidate_topk,
            residual_strength=0.32,
            gate_bias=-1.70,
            fidelity_gate_scale=0.16,
            residual_temperature=1.0,
            center_blend=0.45,
            uncertainty_scale=6.0,
            score_calibration_blend=0.14,
            mean_shift_blend=0.25,
            target_candidate_logit=0.0,
            target_candidate_std=1.0,
            std_gate_scale=4.0,
            mean_gate_scale=4.0,
            mean_shift_threshold=0.35,
            min_graph_gate=0.02,
            rate_shift_blend=0.28,
            rate_gate_scale=4.0,
            rate_shift_threshold=0.15,
            max_rate_shift=1.1,
            min_candidate_rate=0.30,
            max_candidate_rate=0.44,
            uncertainty_rate_scale=5.0,
            uncertainty_rate_center=0.55,
            compression_rate_scale=4.0,
            phase_confidence_scale=8.0,
            phase_confidence_center=0.10,
            coherent_rate_blend=0.55,
        )

    def forward(self, batch) -> torch.Tensor:
        edge_index_seq = batch.edge_index_seq
        edge_index_sec = batch.edge_index_sec
        edge_index_long = batch.edge_index_long
        if not self.use_seq_view:
            edge_index_seq = edge_index_seq.new_empty((2, 0))
        if not self.use_sec_view:
            edge_index_sec = edge_index_sec.new_empty((2, 0))
        if not self.use_long_view:
            edge_index_long = edge_index_long.new_empty((2, 0))
        batch.edge_index_seq = edge_index_seq
        batch.edge_index_sec = edge_index_sec
        batch.edge_index_long = edge_index_long

        x = self.input_proj(batch.x)
        states = [x]
        for layer in self.layers:
            x = layer(x, edge_index_seq, edge_index_sec, edge_index_long)
            states.append(x)
        return self.head(states, batch)
