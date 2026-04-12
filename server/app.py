"""
FastAPI server for HiringNegotiationArena.
Exposes /reset, /step, /state endpoints per OpenEnv spec.
"""
from __future__ import annotations
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Dual-import pattern: relative works in-repo, bare works in Docker
try:
    from ..models import HiringAction, HiringObservation, HiringState, StepResult
except ImportError:
    from models import HiringAction, HiringObservation, HiringState, StepResult  # type: ignore

try:
    from .environment import HiringEnvironment
    from .task_configs import TASK_NAMES, TASKS
except ImportError:
    from server.environment import HiringEnvironment  # type: ignore
    from server.task_configs import TASK_NAMES, TASKS  # type: ignore

app = FastAPI(
    title="HiringNegotiationArena",
    description=(
        "OpenEnv environment where an AI agent acts as a hiring manager, "
        "navigating multi-party negotiations with hidden agendas and bias traps."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: HiringEnvironment = HiringEnvironment(task_name="task1_easy")


@app.get("/")
def root():
    return {
        "name": "HiringNegotiationArena",
        "version": "1.0.0",
        "tasks": TASK_NAMES,
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "healthy", "service": "hiring-negotiation-arena"}


@app.get("/tasks")
def list_tasks():
    return {
        name: {
            "display_name": cfg["display_name"],
            "description": cfg["description"],
            "difficulty": cfg["difficulty"],
            "max_steps": cfg["max_steps"],
            "expected_score_range": cfg["expected_score_range"],
        }
        for name, cfg in TASKS.items()
    }


@app.post("/reset", response_model=HiringObservation)
def reset(task_name: str = Query(default="task1_easy", description="Task to run")):
    global _env
    if task_name not in TASK_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task_name}'. Available: {TASK_NAMES}")
    _env = HiringEnvironment(task_name=task_name)
    obs = _env.reset()
    return obs


@app.post("/step", response_model=StepResult)
def step(action: HiringAction):
    try:
        result = _env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=HiringState)
def get_state():
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()

@app.get("/score")
def get_score():
    """Return current episode score - required by OpenEnv spec."""
    if _env._state is None:
        return {"score": 0.0, "outcome": None, "done": False}
    state = _env._state
    score = 0.0
    if state.done and state.rewards_per_step:
        score = float(state.rewards_per_step[-1])
    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "outcome": state.outcome,
        "done": state.done,
        "steps": state.step,
        "bias_score": state.bias_detector.bias_score,
        "bias_flags": list(state.bias_detector.flags),
    }
