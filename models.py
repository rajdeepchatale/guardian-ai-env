"""
Pydantic models for GuardianAI's action/observation space.

We expose both the worker AI's visible output AND its internal action log
in the observation. This is intentional — the agent needs to cross-reference
these two to catch hallucinations (e.g., bot says "I checked the shipping
API" but the log shows no API call was made).

I went back and forth on how much context to include in the observation.
Minimal payloads gave the agent nothing to work with, so I ended up
packing in permissions, role context, and full action logs. It's a lot
of text but the grading signal is way better this way.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class OversightAction(Action):
    """What the guardian agent submits after seeing a worker action.

    Four decisions at once — classification, severity, response, and reasoning.
    The reasoning part is graded too, so the agent can't just guess the label
    and leave the explanation empty.
    """

    decision: str = Field(
        ...,
        description=(
            "Classification of the worker action. One of: "
            "'safe', 'hallucination', 'unauthorized_action', 'data_leakage', "
            "'safety_violation', 'bias', 'scope_creep', 'false_confidence'"
        ),
    )
    severity: str = Field(
        default="none",
        description="Severity level: 'none', 'low', 'medium', 'high', 'critical'",
    )
    response: str = Field(
        default="allow",
        description="Response action: 'allow', 'warn', 'block'",
    )
    reasoning: str = Field(
        default="",
        description=(
            "Explanation for the decision. Must reference specific evidence "
            "from the worker's action log. Graded for quality."
        ),
    )
    suggestion: str = Field(
        default="",
        description="Optional suggestion for what the worker AI should do instead.",
    )


class OversightObservation(Observation):
    """Everything the guardian sees after each step.

    Yeah this is a chunky payload — but the agent genuinely needs all of it.
    Without the permission list it can't tell authorized from unauthorized.
    Without the action log it can't catch hallucinations. Tried trimming
    it down earlier and the learning signal tanked.
    """

    # session context
    session_id: str = Field(default="", description="Current monitoring session ID")
    domain: str = Field(
        default="customer_support",
        description="Domain being monitored: 'customer_support', 'coding', 'data_analysis'",
    )
    worker_role: str = Field(
        default="", description="Description of the Worker AI's assigned role"
    )
    worker_permissions: List[str] = Field(
        default_factory=list,
        description="List of actions the Worker AI is authorized to perform",
    )
    task_context: str = Field(
        default="", description="What task the Worker AI is currently working on"
    )

    # current worker action to evaluate
    worker_action_text: str = Field(
        default="",
        description="What the Worker AI said or did (visible output)",
    )
    worker_action_log: str = Field(
        default="",
        description=(
            "Internal action log showing API calls, DB queries, etc. "
            "Cross-reference with action_text to detect hallucinations."
        ),
    )
    action_index: int = Field(
        default=0, description="Index of current action in the sequence (0-based)"
    )

    # feedback on the agent's last oversight decision
    feedback: str = Field(default="", description="Detailed grading feedback")
    diagnostics: List[Dict] = Field(
        default_factory=list,
        description="Structured diagnostic feedback: type, severity, message, suggestion",
    )

    # episode progress
    steps_remaining: int = Field(default=0, description="Actions left to evaluate")
    current_score: float = Field(default=0.0, description="Cumulative score so far (0.0-1.0)")
    history: List[Dict] = Field(
        default_factory=list,
        description="Previous oversight decisions and their scores",
    )
