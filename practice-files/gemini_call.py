# This file will test the query to gemini using api keys safely stored in
# env file. Using google SDK and as well as raw http request

# imports
import os
import requests
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()
MY_SECRET_KEY = os.getenv("GEMINI_API_KEY")

if not MY_SECRET_KEY:
    print(f"Error: google_api_key is could not be loaded from the .env file.")
    exit(1)
MY_SECRET_KEY = MY_SECRET_KEY.strip()

client = genai.Client(api_key=MY_SECRET_KEY)

sdk_response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Teach me tensors in just 3 sentences."
)

print(f"the SDK response is:  \n{sdk_response.text}\n")
usage = sdk_response.usage_metadata
print(f"Prompt Tokens: {usage.prompt_token_count}")
print(f"Response Tokens: {usage.candidates_token_count}")
print(f"Total Tokens: {usage.total_token_count}")



# raw https call.

url = url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={MY_SECRET_KEY}"

# setting up the payloads the actuall data


data = {
    "contents": [{
        "parts": [{"text": "Give me a great name for a restaurant that sells Italian food in Delhi"}]
    }],
    # Control our token spending
    "generationConfig": {
        "maxOutputTokens": 500,  # capping the response,
        "temperature": 0.7  # for getting cool answers, balancing deterministic  and creativity.
    }

}   


# setting up the headers
headers = {'Content-Type': 'application/json'}

# making the call
raw_response = requests.post(url, headers=headers, json=data)

# now parsing the json response

if raw_response.status_code == 200:
    result = raw_response.json()
    answer = result['candidates'][0]['content']['parts'][0]['text']
    #print(result)
    # usage
    usage = result.get("usageMetadata", {})  # to get usage
    prompt_tokens = usage.get("promptTokenCount")
    candidate_tokens = usage.get("candidatesTokenCount")
    total_tokens = usage.get("totalTokenCount")

    print(f"the raw http call response is: \n{answer}\n")
    print(f"Token Usage -> Prompt: {prompt_tokens}, Response: {candidate_tokens}, Total: {total_tokens}")
else:
    print(f'Error: {raw_response.status_code}: {raw_response.text}')