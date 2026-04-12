import json, requests, os

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://madhuishere-123-hiring-negotiation-arena.hf.space")

# Reset first
obs = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": "task1_easy"}).json()
print("=== RESET OBS KEYS ===")
print(list(obs.keys()))

# Take one step and see full response
action = {
    "action_type": "probe_team_lead",
    "salary_offer": None,
    "message": "What skills are needed?",
    "benefits": None,
    "skill_to_verify": None
}
result = requests.post(f"{ENV_BASE_URL}/step", json=action).json()
print("\n=== STEP RESULT FULL ===")
print(json.dumps(result, indent=2))