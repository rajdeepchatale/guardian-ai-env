# Contributing to GuardianAI

Thanks for your interest in contributing! This project is part of the OpenEnv ecosystem.

## Adding New Scenarios

The easiest way to contribute is adding new monitoring scenarios:

1. Pick a domain (customer_support, coding, data_analysis — or create a new one)
2. Add your scenario to `server/scenarios.py`
3. Each scenario needs:
   - A `MonitoringScenario` with worker role, permissions, and task context
   - A sequence of `WorkerAction` items with ground truth `ActionLabel`
   - At least one "false positive trap" (looks bad but is actually fine)
4. Add the task to `openenv.yaml`
5. Test: `python3 -c "from server.scenarios import ALL_SCENARIOS; print(len(ALL_SCENARIOS))"`

## Adding New Grading Components

If you want to add a new reward component:

1. Edit `server/graders.py`
2. Add your component to the `GradeResult` dataclass
3. Update the weight distribution (must still sum to 1.0)
4. Add anti-cheat checks if applicable

## Running Locally

```bash
# start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# test it works
curl http://localhost:8000/health
```

## Code Style

- Python 3.10+
- Type hints everywhere
- Docstrings on public functions
- Comments explaining *why*, not *what*
