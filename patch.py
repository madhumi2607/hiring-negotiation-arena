import re

with open('inference.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: wrong action name in force_action and history checks
content = content.replace("'consult_team_lead'", "'probe_team_lead'")
content = content.replace('"consult_team_lead"', '"probe_team_lead"')
content = content.replace('action_type": "consult_team_lead', 'action_type": "probe_team_lead')

# Fix 2: correct env_step payload
content = content.replace('json={"action": action}', 'json=action')

# Fix 3: valid actions set
content = content.replace(
    'has_probed = any("probe_candidate" in h or "consult_team_lead" in h for h in history)',
    'has_probed = any("probe_candidate" in h or "probe_team_lead" in h for h in history)'
)

with open('inference.py', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print('Done!')
