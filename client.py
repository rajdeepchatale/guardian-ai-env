"""Client for connecting to the GuardianAI environment server.\n\nPretty thin wrapper — just maps our Pydantic models to the WebSocket\npayload format that openenv expects. Most of the heavy lifting is in\nEnvClient base class.\n"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import OversightAction, OversightObservation


class GuardianAIEnv(
    EnvClient[OversightAction, OversightObservation, State]
):
    """
    WebSocket client for the GuardianAI oversight environment.

    Example usage::

        async with GuardianAIEnv(base_url="http://localhost:8000") as client:
            result = await client.reset()
            result = await client.step(OversightAction(
                decision="safe", severity="none", response="allow",
                reasoning="Standard greeting, no issues."
            ))
    """

    def _step_payload(self, action: OversightAction) -> Dict:
        return {
            "decision": action.decision,
            "severity": action.severity,
            "response": action.response,
            "reasoning": action.reasoning,
            "suggestion": action.suggestion,
        }

    def _parse_result(self, payload: Dict) -> StepResult[OversightObservation]:
        obs_data = payload.get("observation", {})
        observation = OversightObservation(
            session_id=obs_data.get("session_id", ""),
            domain=obs_data.get("domain", "customer_support"),
            worker_role=obs_data.get("worker_role", ""),
            worker_permissions=obs_data.get("worker_permissions", []),
            task_context=obs_data.get("task_context", ""),
            worker_action_text=obs_data.get("worker_action_text", ""),
            worker_action_log=obs_data.get("worker_action_log", ""),
            action_index=obs_data.get("action_index", 0),
            feedback=obs_data.get("feedback", ""),
            diagnostics=obs_data.get("diagnostics", []),
            steps_remaining=obs_data.get("steps_remaining", 0),
            current_score=obs_data.get("current_score", 0.0),
            history=obs_data.get("history", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
