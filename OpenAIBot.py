import openai
import urllib.request
import urllib3
import json
import os
import ssl
from dataclasses import dataclass
import streamlit as st
import time
import math

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)

@dataclass
class Turn:
    speaker: str
    msg: str

BOT_SPEAKER = "Assistant"
USER_SPEAKER = "User"
GPT35_MODELS = {'text-davinci-002','code-davinci-002'}

http = urllib3.PoolManager()

def nchars_leq_ntokens_approx(maxTokens):
    #returns a number of characters very likely to correspond <= maxTokens
    sqrt_margin = 0.5
    lin_margin = 1.010175047 #= e - 1.001 - sqrt(1 - sqrt_margin) #ensures return 1 when maxTokens=1
    return max( 0, int(maxTokens*math.exp(1) - lin_margin - math.sqrt(max(0,maxTokens - sqrt_margin) ) )) 

def truncate_text_to_maxTokens_approx(text, maxTokens):
    #returns a truncation of text to make it (likely) fit within a token limit
    #So the output string is very likely to have <= maxTokens, no guarantees though.
    char_index = min( len(text), nchars_leq_ntokens_approx(maxTokens) )
    return text[:char_index]


class OAIChatBot:
    def __init__(
        self, 
        base_prompt, 
        oai_key, 
        oai_model, 
        tprompt_uri, 
        tprompt_key, 
        max_context=5,
        num_tprompt_docs=2,
        max_tokens=500,
        temperature=0.,
        top_p=1.,
        query_history = 1
        ):
        self.base_prompt = base_prompt
        openai.api_key = oai_key
        self.oai_model = oai_model
        self.tprompt_uri = tprompt_uri
        self.tprompt_key = tprompt_key
        self.turns = []
        self.user_texts = []
        self.generated_texts = []
        self.max_context = max_context
        self.chatgpt_uri = ''
        self.chatgpt_key = ''
        self.num_tprompt_docs = num_tprompt_docs
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.query_history = query_history

        # if not self.chatgpt_uri or not self.chatgpt_key:
            # raise Exception("Pleae provide uri/key for chat gpt using env variables: CHATGPT_URI, CHATGPT_KEY")
            # with open("../chatgpt_config.json") as rf:
            #     chatgpt_config = json.load(rf)
            #     self.chatgpt_uri = chatgpt_config["uri"]
            #     self.chatgpt_key = chatgpt_config["api-key"]



    def reset(self):
        self.turns = []
        self.user_texts = []
        self.generated_texts = []

    def _fetch_from_tprompt(self, input_prompt):
        headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + self.tprompt_key),
                   'azureml-model-deployment': 'blue'}
        # Make the request and display the response and logs
        data = {
            "inputs": [input_prompt],
            "topK": self.num_tprompt_docs,
            "instruct": "RANDOM",
            "input_delimiter": "input:",
            "output_delimiter": "output:",
            "batch_size": 1
        }
        body = str.encode(json.dumps(data))
        req = urllib.request.Request(self.tprompt_uri, body, headers)

        result = "N/A"
        try:
            response = urllib.request.urlopen(req)
            result = response.read()
        except urllib.error.HTTPError as error:
            raise Exception("Can't read from tprompt endpoint due to", error, result)

        output = json.loads(result)

        prompt_json = output["prompts"][0]
        output_prompt = prompt_json['context'].strip()
        # print (output_prompt)
        fetched_docs = output_prompt.split("input:")[1:-1]
        fetched_docs = [doc.strip().split("output:")[0] for doc in fetched_docs]
        # st.json({"Tprompt Docs": fetched_docs})
        return fetched_docs, output["latency(ms)"], output["vector_search_latency(ms)"]

    def _construct_prompt(self, user_prompt, fetched_doc):
        top_prompt = f"{self.base_prompt}\n\n"
        prompt = ""
        # token_budget = 2048 - (len(featched_doc.split())*2)
        token_budget = 1024 - (len(top_prompt.split())*2)
        self.turns.append(Turn(USER_SPEAKER, user_prompt))
        self.user_texts.append(user_prompt)
        for turn_no in range(len(self.turns)-1, int(max(-1, len(self.turns)-self.max_context-1)), -1):
            this_turn = self.turns[turn_no]
            this_prompt = f"{this_turn.speaker}: {this_turn.msg[:1000]}\n"
            token_budget -= (len(this_prompt.split())*2)
            prompt = this_prompt + prompt
            if token_budget <= 0:
                break
        if not fetched_doc or len(fetched_doc.strip())==0:
            return f"{top_prompt}\n<Conversation>{prompt}{BOT_SPEAKER}:"
        else:
            doc_tokens = fetched_doc.replace("\n", "<NEW_LINE>").split()
            token_budget = 3000 - (len(top_prompt.split())*2) - (len(prompt.split())*2)
            doc_budget = token_budget - 10
            print("Token Budget left for documentation is", token_budget)
            if token_budget > 2:
                fetched_doc = " ".join(doc_tokens[:int(token_budget/2)]).replace("<NEW_LINE>", "\n")
            else:
                
                return f"{top_prompt}\n<Conversation>{prompt}{BOT_SPEAKER}:"
            
            truncated_text = truncate_text_to_maxTokens_approx(fetched_doc, doc_budget)
            return f"{top_prompt}\n<Documentation>{truncated_text}\n<Conversation>{prompt}{BOT_SPEAKER}:"

    def _get_data(self, event_str):
        return event_str.split(":", 1)[1].strip()

    def _get_sse_events(self, response_stream):
        "Hang on to stream until we reach the end."
        "https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events"
        
        event = b''
        for s in response_stream:
            for line in s.splitlines(True):
                event += line
                if event.endswith(b'\n\n'):
                    yield self._get_data(event.decode("utf-8").strip())
                    event = b''

    def _get_oai_completion(self, prompt):
        stashed_msg = ""
        
        if self.oai_model in GPT35_MODELS:
            print("********** In oai completion")
            # try: 
            response = openai.Completion.create(
                engine=self.oai_model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["#"]
            )
            print("Debugger 1: ", response)
            yield response.choices[0].text
            # except Exception as e:
            #     print(f"Error generating completion: {e}") 
        # elif self.oai_model == "chatgpt":
            # uri = self.chatgpt_uri
            # api_key = self.chatgpt_key
            # headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key),
            #            'ocp-apim-subscription-key': api_key,
            #            'api-key': api_key
            #         }
            # input_dict = dict()
            # input_dict["prompt"] = prompt
            # input_dict["temperature"] = self.temperature
            # input_dict["top_p"] = self.top_p
            # input_dict["max_tokens"] = self.max_tokens
            # input_dict["stream"] = True
            # response = http.request('POST', uri, headers=headers, body=json.dumps(input_dict), preload_content=False)

            # for event in self._get_sse_events(response.stream()):
            #     try:
            #         event_json = json.loads(event)
            #     except:
            #         print("GPT query Failed due to  "+ event)
            #         event_json = {}
            #     resp_json = event_json
            #     if "choices" in resp_json and len(resp_json["choices"]) > 0:
            #         text_response = resp_json["choices"][0]['text']
            #         if len(text_response) == 0:
            #             if resp_json["choices"][0]['finish_reason'] == 'content_filter':
            #                 yield " The prompt was filtered due to a content filter. Please change your query or try without tprompt."
            #                 return
            #         else:
            #             if USER_SPEAKER in text_response:
            #                 stashed_msg += text_response
            #             else:
            #                 if stashed_msg:
            #                     text_response = stashed_msg + text_response
            #                     stashed_msg = ""
            #                 is_done, bot_msg = self._clean_up_generation(text_response)
            #                 yield bot_msg
            #                 if is_done:
            #                     return

            #     else:
            #         yield "GPT query Failed due to  " + resp_json
        else:
            raise NotImplementedError(f"Doesnot support the {self.oai_model}")

    def _clean_up_generation(self, generated_output):
        user_delimiter = USER_SPEAKER + ":"
        if user_delimiter in generated_output:
            return True, generated_output.split(user_delimiter)[0]
        return False, generated_output

    def get_reply(self, query, is_tprompt=True):
        if is_tprompt:
            fetched_docs, latency, vecsearch_latency = self._fetch_from_tprompt(f"{query}")
            print("************", fetched_docs)
            # if self.query_history  < 2:
            #      fetched_docs, latency, vecsearch_latency = self._fetch_from_tprompt(f"{query}")
            # else:
            #     past_user_qs = "\n".join(self.user_texts[-1-(self.query_history-1):-1])
            #     fetched_docs, latency, vecsearch_latency = self._fetch_from_tprompt(f"{past_user_qs}\n{query}")
        else:
            fetched_docs, latency, vecsearch_latency = [''], -1, -1
        
        fetched_doc = "<DOC>\n".join(fetched_docs)
        oai_query = self._construct_prompt(query, fetched_doc) # use the top match only
        print ("*********oai query: " + oai_query)
        t1 = time.time()
        curr_msg = ""
        for generated_output in self._get_oai_completion(oai_query):
            oai_generation_time_ms = (time.time() - t1) *1000
            curr_msg += generated_output
            turn_msg = Turn(BOT_SPEAKER, curr_msg)
            self.streaming_currently = True
            yield turn_msg, fetched_doc, latency, vecsearch_latency, oai_query, oai_generation_time_ms
        # print ("current msg: " + curr_msg)
        self.generated_texts.append(curr_msg)
        self.turns.append(Turn(BOT_SPEAKER, curr_msg))
        self.streaming_currently = False