import os
import sys
import time
import ipdb
import logging
from termcolor import colored
from datetime import datetime
import openai
import tqdm
from anthropic import Anthropic

# class for calling OpenAI API and handling cache
class GptApi:
    def __init__(self, verbose=False):
        self.client = Anthropic(
    api_key=""  # This is the default and can be omitted
) # in order to suppress all these HTTP INFO log messages

    # answer_id is used for determining if it was the top answer or how deep in the list it was
    def request(self, prompt, model, parse_response, temperature=0, answer_id=-1, cache=None, max_tokens=None):
        request = {"model": model, "temperature": temperature, "prompt": prompt}

        if request in cache and cache[request] is not None and len(cache[request]) > 0:
            answers = cache[request]
        else:
            answers = self.request_api(prompt, model, temperature, max_tokens)
            cache[request] = answers

        # there is no valid answer
        if len(answers) == 0:
            return [{
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": None,
                    "prompt": prompt,
                    "finish_reason": None,
                    "model": model,
                    }]

        parsed_answers = []
        for full_answer in answers:
            finish_reason = full_answer["finish_reason"]
            full_answer = full_answer["answer"]
            answer_id += 1
            answer = parse_response(full_answer)
            if temperature > 0:
                print(f"Answer (t={temperature}): " + colored(answer, "yellow") + " (" + colored(full_answer, "blue") + ")", file=sys.stderr)
            if answer is None:
                continue
            parsed_answers.append(
                {
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": answer,
                    "prompt": prompt,
                    "finish_reason": finish_reason,
                    "model": model,
                }
            )

        # there was no valid answer, increase temperature and try again
        if len(parsed_answers) == 0:
            return self.request(prompt, model, parse_response, temperature=temperature + 1, answer_id=answer_id, cache=cache)

        return parsed_answers

    def request_api(self, prompt, model, temperature=0, max_tokens=None):
        if temperature > 10:
            return []

        while True:
            try:
                response = self.call_api(prompt, model, temperature, max_tokens)
                break
            except Exception as e:
                # response was filtered
                if hasattr(e, 'code'):
                    if e.code == 'content_filter':
                        return []
                    print(e.code, file=sys.stderr)
                if hasattr(e, 'error') and e.error['code'] == 'invalid_model_output':
                    return []

                # frequent error is reaching the API limit
                print(colored("Error, retrying...", "red"), file=sys.stderr)
                print(e, file=sys.stderr)
                time.sleep(1)

        answers = []

        answers.append({
            "answer": response[0]["answer"],
            "finish_reason": response[0]["answer"],
        })

        if len(answers) > 1:
            # remove duplicate answers
            answers = [dict(t) for t in {tuple(d.items()) for d in answers}]

        return answers

    def call_api(self, prompt, model, temperature, max_tokens):
        parameters = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens else 1024,  # Default to 1024 tokens
            "system": "You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.",

            "messages": prompt,
        }

        response = self.client.messages.create(**parameters)
        
        answer = response.content[0].text.strip()  # Extract response correctly

        return [{
            "answer": answer,
            "finish_reason": response.stop_reason,  # Correct key for Claude's API
        }]

            
    def bulk_request(self, df, model, parse_mqm_answer, cache, max_tokens=None):
        answers = []
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df), file=sys.stderr):
            prompt = row["prompt"]
            parsed_answers = self.request(prompt, model, parse_mqm_answer, cache=cache, max_tokens=max_tokens)
            answers += parsed_answers
        print(answers)
        return answers
