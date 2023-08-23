import requests

url = "http://127.0.0.1:8000/model"

headers = {"Content-Type": "application/json"}
data = {"prompt" : "你好", "temperature" : 0.7}
# print(data)
response = requests.post(url, headers=headers, json=data)
# print(response.text)
result = response.json()
print(result)