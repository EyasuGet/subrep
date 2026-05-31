"""
Skill Selector — strategy layer for choosing which skill to execute.

Sits on top of SkillLibrary and provides different selection strategies:

    Stage 3-4:   select_random()  — uniform random baseline
    Stage 5: select_by_payoff()   — greedy via SkillGenerator
    Stage 6: select_by_mdn()      — MDN-weighted contextual
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from .skill_library import SkillLibrary
from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_selection import alpha_to_mean_weights, select_best_candidate
from utils.mdn_context import build_mdn_context_from_entry


class SkillSelector:
    """
    Choose a skill from the library using a pluggable strategy.

    Attributes:
        library:   The SkillLibrary to select from.
        generator: Optional SkillGenerator for payoff-based selection (Stage 5).
        mdn:       Optional MotiveDecompositionNetwork for MDN selection (Stage 6).
        _rng:      Isolated random state — uses numpy's Generator API so
                   selections are reproducible with the same seed and never
                   interfere with other random generators in the pipeline.
    """

    def __init__(
        self,
        library: SkillLibrary,
        generator=None,
        mdn=None,
        seed: int = 42,
    ) -> None:
        """
        Initialize the selector.
        """
        self.library = library
        self.generator = generator
        self.mdn = mdn
        self._rng = np.random.default_rng(seed)


    def select_random(self, obs: np.ndarray) -> Optional[str]:
        """
        Uniformly random selection from all admitted skills.

        Args:
            obs: Current environment observation.
                 present for interface consistency with other selectors.
        """
        skills = self.library.get_admitted_skills()

        if not skills:
            return None

        # Build list of skill IDs then pick one uniformly at random.
        skill_ids = [s.skill_id for s in skills]
        idx = self._rng.integers(0, len(skill_ids))
        return skill_ids[idx]

    def select_by_payoff(self, obs: np.ndarray) -> Optional[str]:
        """
        Select the skill with the highest Generator-predicted payoff.

        Uses the SkillGenerator to predict expected payoff for the current
        observation, then picks the library skill whose certified delta_r
        is highest. The Generator prediction provides a confidence signal
        that the environment state is favorable for skill execution.

        The scoring formula per skill is simply:

            score = delta_r

        i.e. greedy on the certified payoff improvement. The Generator is
        used as a gate: if its predicted payoff is negative (the current
        state looks unpromising), we fall back to random selection.

        Args:
            obs: Current environment observation (8D for LunarLander).

        Returns:
            skill_id of the best skill, or None if the library is empty.

        Raises:
            ValueError: If no SkillGenerator was provided at construction.
        """
        if self.generator is None:
            raise ValueError(
                "select_by_payoff() requires a trained SkillGenerator. "
                "Pass generator= to the SkillSelector constructor."
            )

        skills = self.library.get_admitted_skills()
        if not skills:
            return None

        # Use Generator to predict expected outcomes
        obs_tensor = torch.tensor(
            np.asarray(obs, dtype=np.float32), dtype=torch.float32
        )
        with torch.no_grad():
            pred_payoff, _ = self.generator(obs_tensor)

        # If the Generator predicts this state is unpromising (negative payoff),
        # fall back to random selection to maintain exploration.
        if float(pred_payoff.item()) < 0.0:
            return self.select_random(obs)

        # Greedy: pick the skill with the highest certified delta_r.
        best_entry = max(skills, key=lambda e: e.delta_r)
        return best_entry.skill_id

    def select_by_mdn(self, obs: np.ndarray) -> Optional[str]:
        """
        Select the skill with the highest MDN-weighted score.

        Uses the MotiveDecompositionNetwork to predict context-aware
        objective weights w from a 14D context vector, then scores each
        library skill as:

            score = delta_r + w^T * delta_n

        This selects the skill that maximizes the weighted combination
        of payoff improvement and motive improvements, where the weights
        are adapted to the current environment state.

        The context vector for each candidate is built as:
            [obs(8D), delta_r(1D), delta_n(2D), gate_indicator(2D), margin(1D)]

        For the initial MDN query, we use the library's best skill's
        context to get weights, then re-score all candidates under those
        weights.

        Args:
            obs: Current environment observation (8D for LunarLander).

        Returns:
            skill_id of the best skill, or None if the library is empty.

        Raises:
            ValueError: If Generator or MDN were not provided at construction.
        """
        if self.generator is None or self.mdn is None:
            raise ValueError(
                "select_by_mdn() requires both a trained SkillGenerator and "
                "a trained MotiveDecompositionNetwork. Pass generator= and "
                "mdn= to the SkillSelector constructor."
            )

        skills = self.library.get_admitted_skills()
        if not skills:
            return None

        obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)

        # Build context from the first skill to get initial MDN weights.
        # The MDN predicts general trade-off preferences for this observation;
        # individual skill metrics provide tie-breaking through scoring.
        context = build_mdn_context_from_entry(obs_np, skills[0])
        context_tensor = torch.tensor(context, dtype=torch.float32)

        with torch.no_grad():
            alpha, _ = self.mdn.forward_inference(context_tensor)

        weights = alpha_to_mean_weights(alpha.cpu().numpy())

        # Build CandidateSkillRecords from all library entries.
        candidates = tuple(
            CandidateSkillRecord(
                skill_id=entry.skill_id,
                delta_r=entry.delta_r,
                delta_n=entry.delta_n,
                is_certified=True,
                gate_type=entry.gate_type,
                admission_margin=entry.admission_margin,
                epsilon=entry.epsilon,
            )
            for entry in skills
        )

        # Score each candidate as delta_r + w^T * delta_n, pick the best.
        best_id, _ = select_best_candidate(candidates, weights)
        return best_id
