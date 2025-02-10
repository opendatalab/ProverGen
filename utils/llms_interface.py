import os
import abc
import time
from dataclasses import dataclass

from openai import OpenAI
import anthropic


@dataclass(frozen=True)
class LLMResponse:
    prompt_text: str
    response_text: str
    prompt_info: dict
    logprobs: list
    
    
class LanguageModels(abc.ABC):
    """ A pretrained Large language model"""

    @abc.abstractmethod
    def completion(self, prompt: str) -> LLMResponse:
        raise NotImplementedError("Override me!")
    

class GPTChatModel(LanguageModels):

    def __init__(self, args):
        self.model_name = args.model_name
        self.temperature = args.temperature
        self.max_new_tokens = args.max_new_tokens
        
        self.gpt = OpenAI()
        self.err_cnt = 0

    def completion(self, prompt: list) -> LLMResponse:
        while True:
            api_flag = False
            try:
                completion = self.gpt.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens
                )
                result = completion.choices[0].message.content
                api_flag = True
            except: 
                self.err_cnt += 1
                if api_flag == False:
                    print(f"API error occured, wait for 10 seconds. Error count: {self.err_cnt}")
                time.sleep(10)

            if api_flag:
                break
        return self._raw_to_llm_response(model_response=result, prompt_text=prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)

    @staticmethod
    def _raw_to_llm_response(model_response, prompt_text: str, max_new_tokens: int, temperature: float) -> LLMResponse:
        prompt_info = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature
        }

        return LLMResponse(prompt_text=prompt_text, response_text=model_response, prompt_info=prompt_info, logprobs=[])


class ClaudeModel(LanguageModels):
    
    def __init__(self, args) -> None:
        self.model_name = args.model_name
        self.temperature = args.temperature
        self.max_new_tokens = args.max_new_tokens
        
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.err_cnt = 0
        
    def completion(self, prompt: list) -> LLMResponse:
        while True:
            api_flag = False
            try:
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_new_tokens,
                    messages=prompt,
                    system=system_prompt['content']
                )
                result = message.content[0].text
                api_flag = True
            except:
                self.err_cnt += 1
                if api_flag == False:
                    print(f"API error occured, wait for 3 seconds. Error count: {self.err_cnt}")
                time.sleep(3)

            if api_flag:
                break
        
        return self._raw_to_llm_response(model_response=result, prompt_text=prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
    
    @staticmethod
    def _raw_to_llm_response(model_response,
                             prompt_text: str,
                             max_new_tokens: int,
                             temperature: float) -> LLMResponse:
        prompt_info = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature
        }

        return LLMResponse(prompt_text=prompt_text, response_text=model_response, prompt_info=prompt_info, logprobs=[])


class LocalModel(LanguageModels):
    
    def __init__(self, args) -> None:
        self.model_name = args.model_name
        self.temperature = args.temperature
        self.max_new_tokens = args.max_new_tokens
        
        self.model = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url
        )
    
    def completion(self, prompt: list) -> LLMResponse:
        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens
        )
        result = completion.choices[0].message.content

        return self._raw_to_llm_response(model_response=result, prompt_text=prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
    
    @staticmethod
    def _raw_to_llm_response(model_response, prompt_text: str, max_new_tokens: int, temperature: float) -> LLMResponse:
        prompt_info = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature
        }
        
        return LLMResponse(prompt_text=prompt_text, response_text=model_response, prompt_info=prompt_info, logprobs=[])
        

class DeepseekModel(LanguageModels):
    
    def __init__(self, args) -> None:
        self.model_name = args.model_name
        
        self.model = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url
        )
    
    def completion(self, prompt: list) -> LLMResponse:
        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=prompt,
        )
        result = completion.choices[0].message.content

        return self._raw_to_llm_response(model_response=result, prompt_text=prompt, max_new_tokens=-1, temperature=-1)
    
    @staticmethod
    def _raw_to_llm_response(model_response, prompt_text: str, max_new_tokens: int, temperature: float) -> LLMResponse:
        prompt_info = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature
        }
        
        return LLMResponse(prompt_text=prompt_text, response_text=model_response, prompt_info=prompt_info, logprobs=[])


class O1Model(LanguageModels):
    
    def __init__(self, args) -> None:
        self.model_name = args.model_name
        
        self.model = OpenAI()
    
    def completion(self, prompt: list) -> LLMResponse:
        system_message = prompt[0]['content']
        prompt.pop(0)
        prompt[0]['content'] = f"{system_message}\n\n" + prompt[0]['content']

        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=prompt,
        )
        result = completion.choices[0].message.content

        return self._raw_to_llm_response(model_response=result, prompt_text=prompt, max_new_tokens=-1, temperature=-1)
    
    @staticmethod
    def _raw_to_llm_response(model_response, prompt_text: str, max_new_tokens: int, temperature: float) -> LLMResponse:
        prompt_info = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature
        }
        
        return LLMResponse(prompt_text=prompt_text, response_text=model_response, prompt_info=prompt_info, logprobs=[])


class LanguageModelInterface:

    def __init__(self, args) -> None:
        self.model_name = args.model_name

        if 'gpt-4' in self.model_name.lower():
            self.model = GPTChatModel(args)
        elif 'claude-3' in self.model_name.lower():
            self.model = ClaudeModel(args)
        elif 'deepseek' in self.model_name.lower():
            self.model = DeepseekModel(args)
        elif self.model_name in ["o1-preview-2024-09-12", "o1-preview"]:
            self.model = O1Model(args)
        else:
            self.model = LocalModel(args)

    def completion(self, prompt: list) -> LLMResponse:
        return self.model.completion(prompt)

