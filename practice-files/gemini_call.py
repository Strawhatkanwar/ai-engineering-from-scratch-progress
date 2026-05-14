# This file will test the query to gemini using api keys safely stored in
# env file. Using google SDK and as well as raw http request

# imports
import os
import requests
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Teach me tensors in just 3 sentences."
)

print(f"the SDK response is {response.text}")
usage = response.usage_metadata
print(f"Prompt Tokens: {usage.prompt_token_count}")
print(f"Response Tokens: {usage.candidates_token_count}")
print(f"Total Tokens: {usage.total_token_count}")



# raw https call.

# api_key = os.getenv("GOOGLE_API_KEY")
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"

# setting up the payloads the actuall data

data = {
    "contents": [{
        "parts": [{"text": "Give me a great name for a shop that sells Italian food in Delhi"}]
    }]
}

# setting up the headers
headers = {'Content-Type': 'application/json',
           'x-goog-api-key': os.getenv("GOOGLE_API_KEY")}  # calling directly

# making the call
response = requests.post(url, headers=headers, json=data)

# now parsing the json response

if response.status_code == 200:
    result = response.json()
    answer = result['candidates'][0]['content']['parts'][0]['text']
    #print(result)
    # usage
    usage = result.get("usageMetadata", {})  # to get usage
    prompt_tokens = usage.get("promptTokenCount")
    candidate_tokens = usage.get("candidatesTokensCount")
    total_tokens = usage.get("totalTokenCount")

    print(f"the raw http call response is {answer}")
    print(f"Token Usage -> Prompt: {prompt_tokens}, Response: {candidate_tokens}, Total: {total_tokens}")
else:
    print(f'Error: {response.status_code}: {response.text}')