import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import requests

BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")


def is_server_up():
    try:
        return requests.get(f"{BASE_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not is_server_up(), reason="Server not running")
class TestFullEpisodeLoop:

    def test_reset_returns_observation(self):
        r = requests.post(f"{BASE_URL}/reset", params={"task_name": "task1_easy"})
        assert r.status_code == 200
        obs = r.json()
        assert obs["step"] == 0
        assert obs["task_name"] == "task1_easy"
        assert "candidate_name" in obs
        assert "required_skills" in obs
        assert obs["episode_done"] == False

    def test_step_probe_returns_reward(self):
        requests.post(f"{BASE_URL}/reset", params={"task_name": "task1_easy"})
        r = requests.post(f"{BASE_URL}/step", json={
            "action_type": "probe_candidate",
            "action_data": {"question": "Do you have other offers?"}
        })
        assert r.status_code == 200
        result = r.json()
        assert result["reward"] == 0.05
        assert result["done"] == False
        assert "observation" in result

    def test_full_episode_task1(self):
        requests.post(f"{BASE_URL}/reset", params={"task_name": "task1_easy"})
        actions = [
            {"action_type": "probe_team_lead", "action_data": {"topic": "required skills"}},
            {"action_type": "probe_candidate", "action_data": {"question": "salary expectations?"}},
            {"action_type": "check_budget", "action_data": {"proposed_salary": 110000, "justification": "market rate"}},
            {"action_type": "make_offer", "action_data": {"salary": 110000, "title": "Engineer", "start_date": "2025-07-01"}},
        ]
        rewards = []
        done = False
        for action in actions:
            if done:
                break
            r = requests.post(f"{BASE_URL}/step", json=action)
            result = r.json()
            rewards.append(result["reward"])
            done = result["done"]
        assert done == True
        assert sum(rewards) > 0.5

    def test_score_in_range(self):
        requests.post(f"{BASE_URL}/reset", params={"task_name": "task1_easy"})
        requests.post(f"{BASE_URL}/step", json={
            "action_type": "probe_team_lead",
            "action_data": {"topic": "skills"}
        })
        requests.post(f"{BASE_URL}/step", json={
            "action_type": "make_offer",
            "action_data": {"salary": 110000, "title": "Engineer", "start_date": "2025-07-01"}
        })
        r = requests.get(f"{BASE_URL}/state")
        assert r.status_code == 200
        state = r.json()
        assert 0.0 <= state.get("cumulative_reward", 0) <= 10.0

    def test_all_5_tasks_reset(self):
        tasks = ["task1_easy", "task2_medium", "task3_hard", "task4_crisis", "task5_marathon"]
        for task in tasks:
            r = requests.post(f"{BASE_URL}/reset", params={"task_name": task})
            assert r.status_code == 200, f"Reset failed for {task}"
            obs = r.json()
            assert obs["task_name"] == task, f"Wrong task returned for {task}"

    def test_bias_rejection_penalised(self):
        requests.post(f"{BASE_URL}/reset", params={"task_name": "task1_easy"})
        requests.post(f"{BASE_URL}/step", json={
            "action_type": "reject_candidate",
            "action_data": {"reason": "candidate did not attend a top-tier university"}
        })
        r = requests.get(f"{BASE_URL}/state")
        state = r.json()
        assert state.get("done") == True

    def test_invalid_action_returns_error(self):
        requests.post(f"{BASE_URL}/reset", params={"task_name": "task1_easy"})
        r = requests.post(f"{BASE_URL}/step", json={
            "action_type": "invalid_action_xyz",
            "action_data": {}
        })
        result = r.json()
        obs = result.get("observation", {})
        assert obs.get("last_action_error") is not None

    def test_episode_ends_at_max_steps(self):
        requests.post(f"{BASE_URL}/reset", params={"task_name": "task4_crisis"})
        done = False
        steps = 0
        while not done and steps < 20:
            r = requests.post(f"{BASE_URL}/step", json={
                "action_type": "probe_candidate",
                "action_data": {"question": "tell me more"}
            })
            result = r.json()
            done = result["done"]
            steps += 1
        assert done == True
