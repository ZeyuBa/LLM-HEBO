"""The utility functions for prompting GPTmodels.""" 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import os
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
import asyncio
from openai import AsyncOpenAI


client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_retries=5,
    timeout=30
)

import asyncio
import openai
async def call_openai_server_single_prompt(prompt, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8):
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
      chat_completion = await client.chat.completions.create(**param_dict)
      return chat_completion.choices[0].message.content
    except openai.APIConnectionError as e:
      # When the server could not be reached
      print("The server could not be reached.")
      print(e.__cause__)  # an underlying Exception, likely raised within httpx.
      await asyncio.sleep(5)  # Wait a bit before retrying
      return await call_openai_server_single_prompt(prompt, max_decode_steps=max_decode_steps, temperature=temperature)
    except openai.RateLimitError as e:
      # When a rate limit is exceeded
      print("A 429 status code was received; we should back off a bit.")
      await asyncio.sleep(30)  # Adjust back-off time as needed
      return await call_openai_server_single_prompt(prompt, max_decode_steps=max_decode_steps, temperature=temperature)
    except openai.APIStatusError as e:
      # For non-200 HTTP status codes
      print("Another non-200-range status code was received:")
      print(f"Status Code: {e.status_code}")
      print(f"Response: {e.response}")
      await asyncio.sleep(10)  # Wait a bit before retrying
      return await call_openai_server_single_prompt(prompt, max_decode_steps=max_decode_steps, temperature=temperature)
    except OSError as e:
      retry_time = 5  # Adjust the retry time as needed
      print(
          f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
        )
      await asyncio.sleep(5)  # Generic short wait before retry
      return await call_openai_server_single_prompt(prompt, max_decode_steps=max_decode_steps, temperature=temperature)

async def call_openai_server_func(inputs, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8):
    """Asynchronously calls OpenAI server with a list of input strings and returns their outputs."""
    if isinstance(inputs, str):
        inputs = [inputs]
    tasks = [call_openai_server_single_prompt(input_str, model, max_decode_steps, temperature) for input_str in inputs]
    outputs = await asyncio.gather(*tasks)
    return outputs

# Example usage
async def main():
    responses = await call_openai_server_func(["Hello, world!"], "gpt-3.5-turbo", 20, 0.8)
    return (responses)

if __name__ == "__main__":
    print(asyncio.run(main()))