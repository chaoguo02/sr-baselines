import os, json, requests, sys

api_key = "sk-a4c8d17b5eba495e8e6cca04804f4320"

# Test 1: DashScope compatible-mode endpoint
print("=== Test 1: DashScope compatible-mode ===")
headers = {
    "Authorization": "Bearer " + api_key,
    "Content-Type": "application/json"
}
payload = {
    "model": "qwen3.5-plus",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
}
r = requests.post(
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    json=payload, headers=headers, timeout=30
)
print(f"Status: {r.status_code}")
print(f"Response: {r.text[:500]}")
print()

# Test 2: Use openai library directly (as ICSR does)
print("=== Test 2: openai.Client() ===")
import openai
print(f"OpenAI version: {openai.__version__}")
client = openai.Client(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
print(f"Client base_url: {client.base_url}")
try:
    resp = client.chat.completions.create(
        model="qwen3.5-plus",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    print("SUCCESS: " + resp.choices[0].message.content)
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
print()

# Test 3: Try a different model name
print("=== Test 3: qwen-max ===")
client2 = openai.Client(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
try:
    resp = client2.chat.completions.create(
        model="qwen-max",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    print("SUCCESS: " + resp.choices[0].message.content)
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
print()

# Test 4: Check with standard DashScope endpoint
print("=== Test 4: Standard DashScope API ===")
headers2 = {
    "Authorization": "Bearer " + api_key,
    "Content-Type": "application/json"
}
payload2 = {
    "model": "qwen3.5-plus",
    "input": {"messages": [{"role": "user", "content": "Say hello"}]},
    "parameters": {"max_tokens": 10}
}
r3 = requests.post(
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
    json=payload2, headers=headers2, timeout=30
)
print(f"Status: {r3.status_code}")
print(f"Response: {r3.text[:500]}")
