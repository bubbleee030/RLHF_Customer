import os
import requests
API_URL = "https://your-argilla-server.hf.space"
API_KEY = os.environ["ARGILLA_API_KEY"]  # required; no default
headers = {"X-Argilla-API-Key": API_KEY}
r = requests.get(f"{API_URL}/api/v1/me/datasets", headers=headers)
print(r.status_code, r.text[:200])

r2 = requests.get(f"{API_URL}/api/v1/datasets", headers=headers)
print(r2.status_code, r2.text[:200])

r3 = requests.get(f"{API_URL}/api/v1/workspaces", headers=headers)
print(r3.status_code, r3.text[:200])
