"""Dreaming Cycle module - autonomous prompt optimization and memory review.

This module enables self-evolution: system analyzes prompt effectiveness,
identifies underperforming templates, and proposes improvements for human approval.
Also performs daily memory review to detect and resolve conflicts in semantic memory.

Components:
- analyzer: Calculate metrics from prompt_audits and user_feedback
- proposer: Generate improved prompt templates using LLM
- approval: Discord UI components for human-in-the-loop approval workflow
- memory_review: Analyze semantic memory for conflicts and propose resolutions
"""

from lattice.dreaming.analyzer import PromptMetrics, analyze_prompt_effectiveness
from lattice.dreaming.approval import TemplateComparisonView
from lattice.dreaming.memory_review import (
    ConflictResolution,
    apply_conflict_resolution,
    get_memories_by_subject,
    get_subjects_for_review,
    run_memory_review,
)
from lattice.dreaming.proposer import OptimizationProposal, propose_optimization


__all__ = [
    "OptimizationProposal",
    "PromptMetrics",
    "TemplateComparisonView",
    "ConflictResolution",
    "analyze_prompt_effectiveness",
    "propose_optimization",
    "run_memory_review",
    "apply_conflict_resolution",
    "get_subjects_for_review",
    "get_memories_by_subject",
]
