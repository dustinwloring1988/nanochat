"""
Curriculum learning support for staged training with dynamic hyperparameters.

This module provides:
- CurriculumStage: Definition of a training stage (token ratio, context range, source weights)
- CurriculumScheduler: Manages stage transitions and interpolation
- Context length scheduling within stages
- Source weight scheduling across stages
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional


@dataclass
class CurriculumStage:
    """Definition of a single training stage in the curriculum."""

    name: str
    token_ratio: float  # Fraction of total training tokens for this stage
    context_range: Tuple[int, int]  # (min_context, max_context) for this stage
    source_weights: Dict[str, float]  # Sampling weights for each data source
    subset_weights: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )  # Per-source subset weights

    def __post_init__(self):
        """Validate stage configuration."""
        assert (
            0.0 < self.token_ratio <= 1.0
        ), f"token_ratio must be in (0, 1], got {self.token_ratio}"
        assert (
            self.context_range[0] > 0
        ), f"min context must be positive, got {self.context_range[0]}"
        assert (
            self.context_range[0] <= self.context_range[1]
        ), f"min context ({self.context_range[0]}) must be <= max context ({self.context_range[1]})"

        # Validate source weights sum to approximately 1.0
        total_weight = sum(self.source_weights.values())
        assert (
            0.99 <= total_weight <= 1.01
        ), f"Source weights must sum to ~1.0, got {total_weight} for stage '{self.name}'"

        # Validate subset weights if provided
        for source, subset_weights in self.subset_weights.items():
            if subset_weights:
                total_subset_weight = sum(subset_weights.values())
                assert (
                    0.99 <= total_subset_weight <= 1.01
                ), f"Subset weights for {source} must sum to ~1.0, got {total_subset_weight}"

    def get_context_length(self, progress: float) -> int:
        """
        Get interpolated context length for this stage.

        Args:
            progress: Progress through this stage (0.0 to 1.0)

        Returns:
            Interpolated context length
        """
        progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
        min_ctx, max_ctx = self.context_range

        if min_ctx == max_ctx:
            return min_ctx

        # Linear interpolation
        ctx = min_ctx + (max_ctx - min_ctx) * progress
        return int(ctx)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class CurriculumScheduler:
    """
    Manages curriculum stages and provides scheduling information for training.

    The scheduler:
    - Tracks which stage we're in based on training progress
    - Interpolates context length within stages
    - Provides current source sampling weights
    - Handles stage transitions
    """

    def __init__(
        self,
        stages: List[CurriculumStage],
        total_tokens: int,
        total_steps: Optional[int] = None,
    ):
        """
        Initialize curriculum scheduler.

        Args:
            stages: List of curriculum stages (in order)
            total_tokens: Total training tokens across all stages
            total_steps: Total training steps (optional, for step-based scheduling)
        """
        self.stages = stages
        self.total_tokens = total_tokens
        self.total_steps = total_steps

        # Validate stages
        assert len(stages) > 0, "Must provide at least one stage"
        total_ratio = sum(stage.token_ratio for stage in stages)
        assert (
            0.99 <= total_ratio <= 1.01
        ), f"Stage token ratios must sum to ~1.0, got {total_ratio}"

        self.stage_boundaries = []
        cumulative_ratio = 0.0
        for stage in stages:
            cumulative_ratio += stage.token_ratio
            self.stage_boundaries.append(round(cumulative_ratio * total_tokens))
        self.stage_boundaries[-1] = total_tokens
        if any(
            right <= left
            for left, right in zip(self.stage_boundaries, self.stage_boundaries[1:])
        ):
            raise ValueError("Token budget is too small to give every stage a step")

        if total_steps is not None:
            if total_steps < len(stages):
                raise ValueError("Total steps must give every stage at least one step")
            self.stage_step_boundaries = []
            cumulative_ratio = 0.0
            for stage in stages:
                cumulative_ratio += stage.token_ratio
                self.stage_step_boundaries.append(round(cumulative_ratio * total_steps))
            self.stage_step_boundaries[-1] = total_steps
            if any(
                right <= left
                for left, right in zip(
                    self.stage_step_boundaries, self.stage_step_boundaries[1:]
                )
            ):
                raise ValueError("Step budget is too small to give every stage a step")
        else:
            self.stage_step_boundaries = None

    def get_current_stage_index(
        self, tokens_processed: Optional[int] = None, step: Optional[int] = None
    ) -> int:
        """
        Get the index of the current stage.

        Args:
            tokens_processed: Number of tokens processed so far
            step: Current training step

        Returns:
            Index of current stage (0-indexed)
        """
        if tokens_processed is not None:
            for idx, boundary in enumerate(self.stage_boundaries):
                if tokens_processed < boundary:
                    return idx
            return len(self.stages) - 1  # Last stage

        elif step is not None and self.stage_step_boundaries is not None:
            for idx, boundary in enumerate(self.stage_step_boundaries):
                if step < boundary:
                    return idx
            return len(self.stages) - 1  # Last stage

        else:
            raise ValueError("Must provide either tokens_processed or step")

    def get_stage_progress(
        self, tokens_processed: Optional[int] = None, step: Optional[int] = None
    ) -> Tuple[int, float]:
        """
        Get current stage and progress through that stage.

        Args:
            tokens_processed: Number of tokens processed so far
            step: Current training step

        Returns:
            Tuple of (stage_index, progress) where progress is in [0.0, 1.0]
        """
        stage_idx = self.get_current_stage_index(tokens_processed, step)

        # Calculate progress through current stage
        if tokens_processed is not None:
            stage_start = 0 if stage_idx == 0 else self.stage_boundaries[stage_idx - 1]
            stage_end = self.stage_boundaries[stage_idx]
            stage_tokens = stage_end - stage_start
            tokens_into_stage = tokens_processed - stage_start
            progress = tokens_into_stage / stage_tokens if stage_tokens > 0 else 0.0

        elif step is not None and self.stage_step_boundaries is not None:
            stage_start = (
                0 if stage_idx == 0 else self.stage_step_boundaries[stage_idx - 1]
            )
            stage_end = self.stage_step_boundaries[stage_idx]
            stage_steps = stage_end - stage_start
            steps_into_stage = step - stage_start
            progress = steps_into_stage / stage_steps if stage_steps > 0 else 0.0

        else:
            raise ValueError("Must provide either tokens_processed or step")

        return stage_idx, progress

    def get_context_length(
        self, tokens_processed: Optional[int] = None, step: Optional[int] = None
    ) -> int:
        """
        Get the current context length based on curriculum stage and progress.

        Args:
            tokens_processed: Number of tokens processed so far
            step: Current training step

        Returns:
            Current context length
        """
        stage_idx, progress = self.get_stage_progress(tokens_processed, step)
        stage = self.stages[stage_idx]
        return stage.get_context_length(progress)

    def get_source_weights(
        self, tokens_processed: Optional[int] = None, step: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get current source sampling weights.

        Args:
            tokens_processed: Number of tokens processed so far
            step: Current training step

        Returns:
            Dictionary of source weights
        """
        stage_idx = self.get_current_stage_index(tokens_processed, step)
        stage = self.stages[stage_idx]
        return stage.source_weights.copy()

    def get_subset_weights(
        self,
        source_name: str,
        tokens_processed: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Get current subset sampling weights for a specific source.

        Args:
            source_name: Name of the data source
            tokens_processed: Number of tokens processed so far
            step: Current training step

        Returns:
            Dictionary of subset weights, or None if not specified
        """
        stage_idx = self.get_current_stage_index(tokens_processed, step)
        stage = self.stages[stage_idx]
        return stage.subset_weights.get(source_name, None)

    def get_stage_info(
        self, tokens_processed: Optional[int] = None, step: Optional[int] = None
    ) -> Dict:
        """
        Get comprehensive information about current stage.

        Args:
            tokens_processed: Number of tokens processed so far
            step: Current training step

        Returns:
            Dictionary with stage information
        """
        stage_idx, progress = self.get_stage_progress(tokens_processed, step)
        stage = self.stages[stage_idx]
        context_length = stage.get_context_length(progress)

        # Calculate tokens and steps for current stage
        stage_start_tokens = (
            0 if stage_idx == 0 else self.stage_boundaries[stage_idx - 1]
        )
        stage_end_tokens = self.stage_boundaries[stage_idx]
        stage_total_tokens = stage_end_tokens - stage_start_tokens

        info = {
            "stage_index": stage_idx,
            "stage_name": stage.name,
            "progress": progress,
            "context_length": context_length,
            "context_range": stage.context_range,
            "source_weights": stage.source_weights,
            "stage_total_tokens": stage_total_tokens,
            "stage_start_tokens": stage_start_tokens,
            "stage_end_tokens": stage_end_tokens,
        }

        if self.stage_step_boundaries is not None:
            stage_start_steps = (
                0 if stage_idx == 0 else self.stage_step_boundaries[stage_idx - 1]
            )
            stage_end_steps = self.stage_step_boundaries[stage_idx]
            info.update(
                {
                    "stage_total_steps": stage_end_steps - stage_start_steps,
                    "stage_start_steps": stage_start_steps,
                    "stage_end_steps": stage_end_steps,
                }
            )

        return info

    def is_stage_transition(
        self,
        tokens_processed: Optional[int] = None,
        step: Optional[int] = None,
        prev_tokens: Optional[int] = None,
        prev_step: Optional[int] = None,
    ) -> bool:
        """
        Check if we just transitioned to a new stage.

        Args:
            tokens_processed: Current tokens processed
            step: Current training step
            prev_tokens: Previous tokens processed
            prev_step: Previous training step

        Returns:
            True if we just transitioned stages
        """
        current_idx = self.get_current_stage_index(tokens_processed, step)

        if prev_tokens is not None:
            prev_idx = self.get_current_stage_index(prev_tokens, None)
        elif prev_step is not None:
            prev_idx = self.get_current_stage_index(None, prev_step)
        else:
            return False

        return current_idx != prev_idx

    def print_curriculum_plan(self):
        """Print a summary of the curriculum plan."""
        print("=" * 80)
        print("CURRICULUM PLAN")
        print("=" * 80)
        print(f"Total tokens: {self.total_tokens:,}")
        if self.total_steps:
            print(f"Total steps: {self.total_steps:,}")
        print()

        for idx, stage in enumerate(self.stages):
            stage_start = 0 if idx == 0 else self.stage_boundaries[idx - 1]
            stage_end = self.stage_boundaries[idx]
            stage_tokens = stage_end - stage_start

            print(f"Stage {idx + 1}: {stage.name}")
            print(f"  Tokens: {stage_tokens:,} ({stage.token_ratio*100:.1f}% of total)")
            print(f"  Token range: {stage_start:,} → {stage_end:,}")
            print(f"  Context: {stage.context_range[0]:,} → {stage.context_range[1]:,}")
            print(f"  Sources:")
            for source, weight in stage.source_weights.items():
                print(f"    {source}: {weight*100:.1f}%")
                if source in stage.subset_weights:
                    for subset, subset_weight in stage.subset_weights[source].items():
                        print(f"      ↳ {subset}: {subset_weight*100:.1f}%")

            if self.stage_step_boundaries:
                stage_start_step = (
                    0 if idx == 0 else self.stage_step_boundaries[idx - 1]
                )
                stage_end_step = self.stage_step_boundaries[idx]
                print(f"  Steps: {stage_start_step:,} → {stage_end_step:,}")

            print()

        print("=" * 80)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "total_tokens": self.total_tokens,
            "total_steps": self.total_steps,
            "stage_boundaries": self.stage_boundaries,
            "stage_step_boundaries": self.stage_step_boundaries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CurriculumScheduler":
        """Reconstruct from dictionary."""
        stages = [
            CurriculumStage(
                name=s["name"],
                token_ratio=s["token_ratio"],
                context_range=tuple(s["context_range"]),
                source_weights=s["source_weights"],
                subset_weights=s.get("subset_weights", {}),
            )
            for s in data["stages"]
        ]
        return cls(
            stages=stages,
            total_tokens=data["total_tokens"],
            total_steps=data.get("total_steps"),
        )
