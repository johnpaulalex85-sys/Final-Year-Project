import requests

try:
    response = requests.get("http://localhost:8000/api/describe_scene")
    print(response.status_code)
    print(response.json())
except Exception as e:
    print(e)
