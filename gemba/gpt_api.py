import sys
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from termcolor import colored
from tqdm import tqdm
from anthropic import Anthropic

class GptApi:
    def __init__(self, verbose=False, num_workers=4):
        self.verbose = verbose
        self.num_workers = num_workers
        self.client = Anthropic(
            api_key=""
        )
        # Thread-local storage to create separate clients per thread
        self.thread_local = threading.local()

    def get_client(self):
        """Get a thread-local client to ensure thread safety"""
        if not hasattr(self.thread_local, "client"):
            self.thread_local.client = Anthropic(
                api_key=""
            )
        return self.thread_local.client

    # Single request method (existing functionality)
    def request(self, prompt, model, parse_response, temperature=0, answer_id=-1, cache=None, max_tokens=None):
        request = {"model": model, "temperature": temperature, "prompt": prompt}

        if cache and request in cache and cache[request] is not None and len(cache[request]) > 0:
            answers = cache[request]
        else:
            answers = self.request_api(prompt, model, temperature, max_tokens)
            if cache:
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
            if temperature > 0 and self.verbose:
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

    # Process a single prompt in a worker thread
    def process_single_prompt(self, prompt, model, parse_response, temperature, max_tokens, cache):
        try:
            results = self.request(prompt, model, parse_response, temperature, cache=cache, max_tokens=max_tokens)
            return results
        except Exception as e:
            print(colored(f"Error processing prompt: {e}", "red"), file=sys.stderr)
            return [{
                "temperature": temperature,
                "answer_id": -1,
                "answer": None,
                "prompt": prompt,
                "finish_reason": "error",
                "model": model,
                "error": str(e)
            }]

    def request_api(self, prompt, model, temperature=0, max_tokens=None):
        if temperature > 10:
            return []

        client = self.get_client()
        
        while True:
            try:
                response = self.call_api(prompt, model, temperature, max_tokens, client)
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
                print(colored(f"Error, retrying in 1s: {str(e)}", "red"), file=sys.stderr)
                time.sleep(1)

        answers = []

        answers.append({
            "answer": response[0]["answer"],
            "finish_reason": response[0]["finish_reason"],
        })

        if len(answers) > 1:
            # remove duplicate answers
            answers = [dict(t) for t in {tuple(d.items()) for d in answers}]

        return answers

    def call_api(self, prompt, model, temperature, max_tokens, client=None):
        if client is None:
            client = self.get_client()
            
        parameters = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens else 1024,  # Default to 1024 tokens
            "system": "You are an expert buddhist annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.",
            "messages": prompt,
        }

        response = client.messages.create(**parameters)
        
        answer = response.content[0].text.strip()  # Extract response correctly

        return [{
            "answer": answer,
            "finish_reason": response.stop_reason,  # Correct key for Claude's API
        }]
    
    # Concurrent processing using ThreadPoolExecutor
    def bulk_request(self, df, model, parse_mqm_answer, cache, max_tokens=None, max_concurrent=None):
        """
        Process a dataframe of prompts using concurrent threading
        
        Args:
            df: Dataframe containing prompts
            model: Model to use for generation
            parse_mqm_answer: Function to parse the responses
            cache: Cache to use for storing responses (can be shared between threads)
            max_tokens: Maximum tokens for generation
            max_concurrent: Maximum number of concurrent requests (default: self.num_workers)
            
        Returns:
            List of parsed answers
        """
        all_answers = []
        prompts = df["prompt"].tolist()
        
        if not max_concurrent:
            max_concurrent = self.num_workers
        
        if self.verbose:
            print(f"Processing {len(prompts)} prompts with {max_concurrent} concurrent threads", file=sys.stderr)
        
        # Create a shared progress bar
        pbar = tqdm(total=len(prompts), desc="Processing prompts", file=sys.stderr)
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Create a list to store all the futures
            futures = []
            
            # Submit all tasks to the executor
            for i, prompt in enumerate(prompts):
                future = executor.submit(
                    self.process_single_prompt,
                    prompt, model, parse_mqm_answer, 0, max_tokens, cache
                )
                futures.append(future)
            
            # Process results as they complete
            for future in futures:
                result = future.result()
                all_answers.extend(result)
                pbar.update(1)
        
        pbar.close()
        return all_answers
    
    # Original sequential processing method for comparison
    def bulk_request_sequential(self, df, model, parse_mqm_answer, cache, max_tokens=None):
        """Sequential processing method"""
        answers = []
        prompts = df["prompt"].tolist() if hasattr(df, "prompt") else df
        
        for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Processing sequentially", file=sys.stderr):
            parsed_answers = self.request(prompt, model, parse_mqm_answer, cache=cache, max_tokens=max_tokens)
            answers.extend(parsed_answers)
                
        return answers