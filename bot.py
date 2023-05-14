# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount
from OpenAIBot import OAIChatBot
import requests
from dataclasses import dataclass
import json
# from azure.identity import ManagedIdentityCredential
from azure.identity import DefaultAzureCredential

from azure.keyvault.secrets import SecretClient

@dataclass
class Turn:
    speaker: str
    msg: str

oai_key = '516a05f6bed44ddeb2a6e8a047046ad5'
oai_model = 'gpt-35-turbo'
oai_uri = 'https://augloop-cs-test-scus-shared-open-ai-0.openai.azure.com/openai/deployments/text-davinci-002/completions?api-version=2022-12-01'
# tprompt_uri = 'https://office-1js-tprompt.eastus2.inference.ml.azure.com/score'
# tprompt_key = 'b8Iv5tSwcdKV9EgCECQe7eCgx4NlEue1'
tprompt_key_vault_url = 'https://tprompt-1js-vault.vault.azure.net/'

def get_tprompt_url():
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=tprompt_key_vault_url, credential=credential)

        tprompt_uri = client.get_secret("tprompt-1js-endpoint-url").value
        tprompt_key = client.get_secret("tprompt-1js-endpoint-key").value
        return tprompt_key, tprompt_uri
    except Exception as e:
        print("Error in getting tprompt url ", e)
        return None, None

tprompt_key, tprompt_uri = get_tprompt_url()
bot = OAIChatBot("", oai_key, oai_model, tprompt_uri, tprompt_key)

class MyBot(ActivityHandler):
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.
 
    async def on_message_activity(self, turn_context: TurnContext):
        query = turn_context.activity.text
        fetched_docs, latency, vecsearch_latency = bot._fetch_from_tprompt(query)
        reply = get_oai_completion(fetched_docs, query)
        await turn_context.send_activity(reply)
                
    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome!")

   
def get_oai_completion(base_prompt, user_prompt):

    # Set the OpenAI API endpoint and parameters
    model_engine = "text-davinci-003"
    temperature = 0.9
    max_tokens = 500

    # Set the request headers and body
    headers = {'Content-Type': 'application/json',
                   'api-key' : oai_key }
    data = {
        "model": oai_model,
        "prompt": f"{base_prompt}\n\n{user_prompt}",
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # Make the API call to OpenAI
    response = requests.post(oai_uri, headers=headers, data=json.dumps(data))

    # Extract the resulting text completion from the API response
    text_completion = response.json()["choices"][0]["text"]
    return text_completion
