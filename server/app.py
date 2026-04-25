"""
FastAPI server entry point.

Start with:
    uv run --project . server
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: uv sync"
    ) from e

try:
    from ..models import OversightAction, OversightObservation
    from .guardian_ai_environment import GuardianAIEnvironment
except (ImportError, ModuleNotFoundError):
    from models import OversightAction, OversightObservation
    from server.guardian_ai_environment import GuardianAIEnvironment


app = create_app(
    GuardianAIEnvironment,
    OversightAction,
    OversightObservation,
    env_name="guardian_ai",
    max_concurrent_envs=256,
)


def main():
    """Entry point for `uv run server` and `openenv serve`."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
