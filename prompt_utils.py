"""The utility functions for prompting GPTmodels.""" 
import os
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import asyncio
from openai import AsyncOpenAI,OpenAI

import asyncio
import openai
openai_client_async = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_retries=5,
    timeout=30
)
async def call_openai_server_single_prompt_async(prompt, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8):
    os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'

    """Asynchronously calls OpenAI API for a single prompt and returns the response."""
    try:
      param_dict={
         "messages":[{"role": "user", "content": prompt}],
         "model":model,
         "temperature":temperature,
         "max_tokens":max_decode_steps
      }
      if '```json' in prompt:
        param_dict['response_format']={"type": "json_object"}
      chat_completion = await openai_client_async.chat.completions.create(**param_dict)
      return chat_completion.choices[0].message.content
    except openai.APIConnectionError as e:
      # When the server could not be reached
      print("The server could not be reached.")
      print(e.__cause__)  # an underlying Exception, likely raised within httpx.
      await asyncio.sleep(5)  # Wait a bit before retrying
      return await call_openai_server_single_prompt_async(prompt, max_decode_steps=max_decode_steps, temperature=temperature)
    except openai.RateLimitError as e:
      # When a rate limit is exceeded
      print("A 429 status code was received; we should back off a bit.")
      await asyncio.sleep(30)  # Adjust back-off time as needed
      return await call_openai_server_single_prompt_async(prompt, max_decode_steps=max_decode_steps, temperature=temperature)
    except openai.APIStatusError as e:
      # For non-200 HTTP status codes
      print("Another non-200-range status code was received:")
      print(f"Status Code: {e.status_code}")
      print(f"Response: {e.response}")
      await asyncio.sleep(10)  # Wait a bit before retrying
      return await call_openai_server_single_prompt_async(prompt, max_decode_steps=max_decode_steps, temperature=temperature)
    except OSError as e:
      retry_time = 5  # Adjust the retry time as needed
      print(
          f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
        )
      await asyncio.sleep(5)  # Generic short wait before retry
      return await call_openai_server_single_prompt_async(prompt, max_decode_steps=max_decode_steps, temperature=temperature)

async def call_openai_server_func_async(inputs, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8):

    """Asynchronously calls OpenAI server with a list of input strings and returns their outputs."""
    if isinstance(inputs, str):
        inputs = [inputs]
    tasks = [call_openai_server_single_prompt_async(input_str, model, max_decode_steps, temperature) for input_str in inputs]
    outputs = await asyncio.gather(*tasks)
    return outputs

openai_client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_retries=5,
    timeout=30
)
from time import sleep
def call_openai_server_single_prompt(prompt, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8, top_p=1.0):
    os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'

    try:
      param_dict={
         "messages":[{"role": "user", "content": prompt}],
         "model":model,
         "temperature":temperature,
         "max_tokens":max_decode_steps,
         "top_p":top_p
      }
      if '```json' in prompt:
        param_dict['response_format']={"type": "json_object"}
      chat_completion = openai_client.chat.completions.create(**param_dict)
      return chat_completion.choices[0].message.content
    except openai.APIConnectionError as e:
      # When the server could not be reached
      print("The server could not be reached.")
      print(e.__cause__)  # an underlying Exception, likely raised within httpx.
      sleep(5)  # Wait a bit before retrying
      return call_openai_server_single_prompt(prompt, max_decode_steps=max_decode_steps, temperature=temperature)
    except openai.RateLimitError as e:
      # When a rate limit is exceeded
      print("A 429 status code was received; we should back off a bit.")
      sleep(30)  # Adjust back-off time as needed
      return call_openai_server_single_prompt(prompt, max_decode_steps=max_decode_steps, temperature=temperature)
    except openai.APIStatusError as e:
      # For non-200 HTTP status codes
      print("Another non-200-range status code was received:")
      print(f"Status Code: {e.status_code}")
      print(f"Response: {e.response}")
      sleep(10)  # Wait a bit before retrying
      return call_openai_server_single_prompt(prompt, max_decode_steps=max_decode_steps, temperature=temperature)
    except OSError as e:
      retry_time = 5  # Adjust the retry time as needed
      print(
          f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
        )
      sleep(5)  # Generic short wait before retry
      return call_openai_server_single_prompt(prompt, max_decode_steps=max_decode_steps, temperature=temperature)

def call_openai_server_func(inputs, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8,top_p=1.0):

    """Asynchronously calls OpenAI server with a list of input strings and returns their outputs."""
    if isinstance(inputs, str):
        inputs = [inputs]
    outputs = [call_openai_server_single_prompt(input_str, model, max_decode_steps, temperature,top_p) for input_str in inputs]
    return outputs




import transformers
import torch

async def call_huggingface_single_prompt(prompt, model="meta-llama/Meta-Llama-3-8B-Instruct", max_decode_steps=20, temperature=0.8):

  pipeline = transformers.pipeline(
      "text-generation",
      model=model,
      model_kwargs={"torch_dtype": torch.bfloat16},
      device="cuda",
  )


  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=max_decode_steps,
      eos_token_id=terminators,
      do_sample=True,
      temperature=temperature,
      # top_p=0.9,
  )
  return outputs[0]["generated_text"][len(prompt):]

async def call_huggingface_func(inputs, model="meta-llama/Meta-Llama-3-8B-Instruct", max_decode_steps=20, temperature=0.8):
    """Asynchronously calls local llm with a list of input strings and returns their outputs."""
    if isinstance(inputs, str):
        inputs = [inputs]
    tasks = [call_huggingface_single_prompt(input_str, model, max_decode_steps, temperature) for input_str in inputs]
    outputs = await asyncio.gather(*tasks)
    return outputs
from ollama import AsyncClient,Client
ollama_client=Client(host='http://localhost:11434')
def call_ollama_single_prompt(prompt,model='llama3:8b-instruct-fp16',max_decode_steps=20, temperature=0.8):
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    format = 'json' if '```json' in prompt else ''
    param_dict={
         "messages":[{"role": "user", "content": prompt}],
         "model":model,
        #  "temperature":temperature,
        #  "max_tokens":max_decode_steps,
         "format":format
      }
    response = ollama_client.chat(**param_dict)
    return response['message']['content']
def call_ollama_func(inputs, model="llama3:8b-instruct-fp16", max_decode_steps=20, temperature=0.8):
    if isinstance(inputs, str):
        inputs = [inputs]
    outputs = [call_ollama_single_prompt(input_str, model, max_decode_steps, temperature) for input_str in inputs]
    return outputs
# Example usage
async def main():
    responses = await call_ollama_func(["Hello, world!"]*2, "llama3:8b-instruct-fp16", 20, 0.8)
    return (responses)

if __name__ == "__main__":
    # print(asyncio.run(main()))
   print(call_openai_server_func(["Hello World!"]*2))