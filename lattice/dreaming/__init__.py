"""Dreaming Cycle module - autonomous prompt optimization.

This module enables self-evolution: the system analyzes prompt effectiveness,
identifies underperforming templates, and proposes improvements for human approval.

Components:
- analyzer: Calculate metrics from prompt_audits and user_feedback
- proposer: Generate improved prompt templates using LLM
- approval: Discord UI components for human-in-the-loop approval workflow
"""

from lattice.dreaming.analyzer import PromptMetrics, analyze_prompt_effectiveness
from lattice.dreaming.approval import ProposalApprovalView, ProposalRejectionModal
from lattice.dreaming.proposer import OptimizationProposal, propose_optimization


__all__ = [
    "OptimizationProposal",
    "PromptMetrics",
    "ProposalApprovalView",
    "ProposalRejectionModal",
    "analyze_prompt_effectiveness",
    "propose_optimization",
]
