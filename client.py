"""
client.py -- HiringNegotiationArena OpenEnv Client

Usage:
    from client import HiringEnv, HiringAction

    env = HiringEnv(base_url="http://localhost:7860")
    obs = env.reset("task1_easy")
    result = env.step(HiringAction(action_type="probe_candidate",
                                   action_data={"question": "Do you have other offers?"}))
    print(result.observation.candidate_interest)
    env.close()
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import requests

try:
    from models import HiringAction, HiringObservation, HiringState, StepResult
except ImportError:
    from .models import HiringAction, HiringObservation, HiringState, StepResult  # type: ignore


class HiringEnv:
    """
    HTTP client for HiringNegotiationArena.
    Implements the OpenEnv reset/step/state interface.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "task1_easy") -> HiringObservation:
        """Reset environment and return initial observation."""
        resp = self._session.post(
            f"{self.base_url}/reset",
            params={"task_name": task_name},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return HiringObservation(**resp.json())

    def step(self, action: HiringAction) -> StepResult:
        """Take one action and return (observation, reward, done, info)."""
        resp = self._session.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return StepResult(**resp.json())

    def state(self) -> HiringState:
        """Return full internal state including hidden party info."""
        resp = self._session.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return HiringState(**resp.json())

    def tasks(self) -> Dict[str, Any]:
        """List all available tasks."""
        resp = self._session.get(f"{self.base_url}/tasks", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, str]:
        """Check server health."""
        resp = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the HTTP session."""
        self._session.close()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def probe_candidate(self, question: str) -> StepResult:
        return self.step(HiringAction(
            action_type="probe_candidate",
            action_data={"question": question},
        ))

    def probe_team_lead(self, topic: str) -> StepResult:
        return self.step(HiringAction(
            action_type="probe_team_lead",
            action_data={"topic": topic},
        ))

    def check_budget(self, proposed_salary: float, justification: str = "") -> StepResult:
        return self.step(HiringAction(
            action_type="check_budget",
            action_data={"proposed_salary": proposed_salary, "justification": justification},
        ))

    def make_offer(self, salary: float, title: str, start_date: str = "2025-07-01") -> StepResult:
        return self.step(HiringAction(
            action_type="make_offer",
            action_data={"salary": salary, "title": title, "start_date": start_date},
        ))

    def reject_candidate(self, reason: str) -> StepResult:
        return self.step(HiringAction(
            action_type="reject_candidate",
            action_data={"reason": reason},
        ))

    def extend_deadline(self) -> StepResult:
        return self.step(HiringAction(
            action_type="extend_deadline",
            action_data={},
        ))


# ------------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("HiringNegotiationArena Client Demo")
    print("=" * 40)

    with HiringEnv(base_url="http://localhost:7860") as env:
        print(f"Health: {env.health()}")
        print(f"Tasks: {list(env.tasks().keys())}")
        print()

        for task_name in ["task1_easy", "task2_medium", "task3_hard"]:
            print(f"--- {task_name} ---")
            obs = env.reset(task_name)
            print(f"Role: {obs.role_title} | Budget: ${obs.salary_budget_visible:,.0f}")
            print(f"Candidate: {obs.candidate_name} | Interest: {obs.candidate_interest:.0%}")

            # Probe team lead
            r = env.probe_team_lead("required skills")
            print(f"Team lead: {r.observation.last_responses[0].message[:80]}")

            # Probe candidate
            r = env.probe_candidate("Do you have other offers? What salary do you expect?")
            print(f"Candidate: {r.observation.last_responses[0].message[:80]}")

            # Check budget
            salary = round(obs.candidate_current_salary * 1.18 / 1000) * 1000
            r = env.check_budget(salary, justification="market rate + 18%")
            print(f"Budget: {r.observation.last_responses[0].message[:80]}")

            # Make offer
            r = env.make_offer(salary, obs.role_title)
            print(f"Outcome: {r.observation.outcome} | Score: {r.reward:.3f}")
            print()
