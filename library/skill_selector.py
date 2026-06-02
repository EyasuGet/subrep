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
from utils.mdn_selection import alpha_to_mean_weights
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
        Select the skill with the highest motive-aligned score.

        Uses the SkillGenerator to predict expected (payoff, motives) for the
        current observation. If the predicted payoff is negative (unpromising
        state), falls back to random selection for exploration. Otherwise,
        scores each library skill as:

            score = delta_r + pred_motives^T · delta_n

        where pred_motives is the Generator's predicted motive vector. This
        picks the skill whose certified improvements best align with the
        Generator's expectation of which motives matter in this state.

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

        # Use Generator to predict expected outcomes for this observation.
        obs_tensor = torch.tensor(
            np.asarray(obs, dtype=np.float32), dtype=torch.float32
        )
        with torch.no_grad():
            pred_payoff, pred_motives = self.generator(obs_tensor)

        # If the Generator predicts this state is unpromising (negative payoff),
        # fall back to random selection to maintain exploration.
        if float(pred_payoff.item()) < 0.0:
            return self.select_random(obs)

        # Use Generator's predicted motives as alignment weights to pick the
        # skill whose certified delta_n best matches what the state needs.
        pred_m = pred_motives.cpu().numpy().reshape(-1)
        best_entry = max(
            skills,
            key=lambda e: e.delta_r + float(
                np.dot(pred_m, np.asarray(e.delta_n, dtype=np.float32))
            ),
        )
        return best_entry.skill_id

    def select_by_mdn(self, obs: np.ndarray) -> Optional[str]:
        """
        Select the skill with the highest MDN-weighted score.

        Queries the MotiveDecompositionNetwork **once per candidate skill**
        with a 14D context vector:

            context = [obs(8D), delta_r(1D), delta_n(2D), gate(2D), margin(1D)]

        Each candidate gets its own context → its own Dirichlet α → its own
        simplex weights w. The score for each candidate is:

            score = delta_r + w^T · delta_n

        This means a PDS skill with a tight margin will receive more
        conservative weights than a comfortable CDS skill, even in the
        same environment state.

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

        best_id: Optional[str] = None
        best_score = -float("inf")

        for entry in skills:
            # Each skill gets its OWN 14D context → its OWN weight prediction.
            context = build_mdn_context_from_entry(obs_np, entry)
            context_tensor = torch.tensor(context, dtype=torch.float32)

            with torch.no_grad():
                alpha, _ = self.mdn.forward_inference(context_tensor)

            weights = alpha_to_mean_weights(alpha.cpu().numpy())
            delta_n = np.asarray(entry.delta_n, dtype=np.float32)
            score = float(entry.delta_r + np.dot(weights, delta_n))

            # Tie-break by skill_id for deterministic ordering when scores
            # are equal (e.g. skills with identical certificates).
            if score > best_score or (
                score == best_score
                and (best_id is None or entry.skill_id < best_id)
            ):
                best_id = entry.skill_id
                best_score = score

        return best_id
