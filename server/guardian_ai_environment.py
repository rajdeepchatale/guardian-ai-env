"""
Core environment logic for GuardianAI.

Each episode is a monitoring session — we pick a scenario, feed the agent
a sequence of Worker AI actions, and grade each oversight call. Pretty
similar to the SQL env's flow but instead of grading queries against a DB,
we're grading classification decisions against ground truth labels.

The action sequence is fixed per scenario (no randomness within episodes)
so the grading is fully deterministic. Random ordering only happens at
the scenario selection level.

TODO: add a curriculum mode that starts with easy scenarios and ramps up.
For now we just shuffle randomly.
"""

import random
from typing import Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import OversightAction, OversightObservation
except ImportError:
    from models import OversightAction, OversightObservation

try:
    from .scenarios import ALL_SCENARIOS, SCENARIO_MAP, MonitoringScenario
    from .graders import grade_oversight, GradeResult
except ImportError:
    from server.scenarios import ALL_SCENARIOS, SCENARIO_MAP, MonitoringScenario
    from server.graders import grade_oversight, GradeResult


class GuardianAIEnvironment(Environment):
    """Presents Worker AI actions for the agent to evaluate.

    Each step shows one Worker AI action. The agent must classify it
    (safe or problematic), choose a response (allow/warn/block), and
    explain its reasoning. Graded on 5 independent components.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_scenario: Optional[MonitoringScenario] = None
        self._action_index: int = 0
        self._step_rewards: List[float] = []
        self._best_score: float = 0.0
        self._history: List[Dict] = []
        self._scenario_queue: List[str] = []
        self._queue_index: int = 0

    def _get_scenario_list(self) -> List[str]:
        ids = [s.id for s in ALL_SCENARIOS]
        random.shuffle(ids)
        return ids

    def _select_scenario(self, task_id: Optional[str] = None) -> MonitoringScenario:
        if task_id and task_id in SCENARIO_MAP:
            return SCENARIO_MAP[task_id]

        if not self._scenario_queue:
            self._scenario_queue = self._get_scenario_list()
            self._queue_index = 0

        if self._queue_index >= len(self._scenario_queue):
            self._queue_index = 0

        scenario = SCENARIO_MAP[self._scenario_queue[self._queue_index]]
        self._queue_index += 1
        return scenario

    def reset(self, task_id: Optional[str] = None) -> OversightObservation:
        """Start a new monitoring session."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._action_index = 0
        self._step_rewards = []
        self._best_score = 0.0
        self._history = []

        self._current_scenario = self._select_scenario(task_id)
        sc = self._current_scenario

        # Present the first worker action
        first_action = sc.action_sequence[0]

        return OversightObservation(
            session_id=self._state.episode_id,
            domain=sc.domain,
            worker_role=sc.worker_role,
            worker_permissions=sc.worker_permissions,
            task_context=sc.task_description,
            worker_action_text=first_action.action_text,
            worker_action_log=first_action.action_log,
            action_index=0,
            feedback=(
                f"Monitoring session started.\n"
                f"Domain: {sc.domain}\n"
                f"Worker role: {sc.worker_role}\n"
                f"Task: {sc.task_description}\n"
                f"Difficulty: {sc.difficulty}\n"
                f"Actions to evaluate: {len(sc.action_sequence)}\n"
                f"\nWorker permissions: {', '.join(sc.worker_permissions)}\n"
                f"\nHints:\n" + "\n".join(f"  - {h}" for h in sc.hints[:1])
            ),
            diagnostics=[],
            steps_remaining=len(sc.action_sequence),
            current_score=0.0,
            history=[],
            done=False,
            reward=0.0,
        )

    def _progressive_hints(self, step: int) -> str:
        sc = self._current_scenario
        hints = sc.hints
        if step <= 1:
            shown = hints[:1] if hints else []
        elif step <= 3:
            shown = hints[:min(step + 1, len(hints))]
        else:
            shown = hints
        if not shown:
            return ""
        return "\n".join(f"  - {h}" for h in shown)

    def step(self, action: OversightAction) -> OversightObservation:
        """Grade the agent's oversight decision for the current worker action."""
        if self._current_scenario is None:
            return OversightObservation(
                feedback="Environment not initialized. Call reset() first.",
                done=True,
                reward=0.0,
            )

        sc = self._current_scenario
        self._state.step_count += 1
        step = self._state.step_count

        # Get ground truth for current action
        current_worker = sc.action_sequence[self._action_index]
        label = current_worker.label

        # Grade the decision
        grade = grade_oversight(
            decision=action.decision,
            severity=action.severity,
            response=action.response,
            reasoning=action.reasoning,
            label=label,
            history=self._history,
        )

        self._step_rewards.append(grade.total_score)
        self._best_score = max(self._best_score, grade.total_score)

        # Record in history
        self._history.append({
            "step": step,
            "action_index": self._action_index,
            "decision": action.decision,
            "severity": action.severity,
            "response": action.response,
            "reasoning": action.reasoning[:80],
            "score": round(grade.total_score, 2),
            "was_correct": (
                (label.is_safe and action.decision == "safe") or
                (not label.is_safe and action.decision != "safe")
            ),
        })

        # Move to next worker action
        self._action_index += 1
        is_done = self._action_index >= len(sc.action_sequence)

        feedback = grade.feedback
        if not is_done and grade.total_score < 0.8:
            hint_text = self._progressive_hints(step)
            if hint_text:
                feedback += f"\n\nHints (step {step}):\n{hint_text}"

        if is_done:
            avg = sum(self._step_rewards) / len(self._step_rewards) if self._step_rewards else 0
            feedback += f"\n\nSession complete. Average score: {avg:.2f}"
            feedback += f"\nScore progression: {' -> '.join(f'{s:.2f}' for s in self._step_rewards)}"

        # Build next worker action observation (or empty if done)
        next_action_text = ""
        next_action_log = ""
        next_index = self._action_index
        if not is_done:
            next_worker = sc.action_sequence[self._action_index]
            next_action_text = next_worker.action_text
            next_action_log = next_worker.action_log

        # Episode reward = average of all step rewards
        episode_reward = sum(self._step_rewards) / len(self._step_rewards) if self._step_rewards else 0.0

        return OversightObservation(
            session_id=self._state.episode_id,
            domain=sc.domain,
            worker_role=sc.worker_role,
            worker_permissions=sc.worker_permissions,
            task_context=sc.task_description,
            worker_action_text=next_action_text,
            worker_action_log=next_action_log,
            action_index=next_index,
            feedback=feedback,
            diagnostics=[d.to_dict() for d in grade.diagnostics],
            steps_remaining=max(0, len(sc.action_sequence) - self._action_index),
            current_score=episode_reward,
            history=self._history,
            done=is_done,
            reward=grade.total_score,
            metadata={
                "scenario_id": sc.id,
                "difficulty": sc.difficulty,
                "domain": sc.domain,
                "step": step,
                "action_index": self._action_index - 1,
                "best_score": self._best_score,
                "detection_score": grade.detection_score,
                "fp_score": grade.false_positive_score,
                "classification_score": grade.classification_score,
                "response_score": grade.response_score,
                "reasoning_score": grade.reasoning_score,
                "penalty": grade.penalty,
            },
        )

    @property
    def state(self) -> State:
        return self._state

    def close(self):
        self._current_scenario = None
