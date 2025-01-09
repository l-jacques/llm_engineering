import os
from enum import Enum, auto
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

def load_allbots():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    google_api_key = os.getenv('GOOGLE_API_KEY')
    
    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set")
        
    if anthropic_api_key:
        print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
    else:
        print("Anthropic API Key not set")
    
    if google_api_key:
        print(f"Google API Key exists and begins {google_api_key[:8]}")
    else:
        print("Google API Key not set")
    
    openai = OpenAI()
    
    claude = anthropic.Anthropic()
    
    gemini_via_openai = OpenAI(
        api_key=google_api_key, 
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

    return openai, claude, gemini_via_openai, ollama_via_openai

def formatPrompt(role, content):
    return {"role": role, "content": content}
    
class AI(Enum):
    OPEN_AI = "OPEN_AI"
    CLAUDE = "CLAUDE"
    GEMINI = "GEMINI"
    OLLAMA = "OLLAMA"
    
class AISystem:
    def __init__(self, processor, system_string="", model="", type=AI.OPEN_AI):
        """
        Initialize the ChatSystem with a system string and empty messages list.
        
        :param system_string: Optional initial system string description
        """
        self.processor = processor
        self.system = system_string
        self.model = model
        self.messages = []
        self.type = type

    def callWithTools(self, tools): 
        if self.type == AI.CLAUDE:
            message = self.processor.messages.create(
            model=self.model,
            system=self.system,
            messages=self.messages,
            tools=tools,
            max_tokens=500
            )
            return message.content[0].text
        else:
            self.messages.insert(0,self.system)
            completion = self.processor.chat.completions.create(
            model = self.model,
            messages= self.messages,
            tools=tools
            )
        return completion.choices[0]
    def call(self, message, tools = []):
        self.messages.append(message)
        toSend = self.messages
      
        return self.callWithTools(tools)

    def stream(self, message, usingGradio=False):
        self.messages.append(message)
      
        if self.type == AI.CLAUDE:
            result  = self.processor.messages.stream(
                        model=self.model,
                        system=self.system,
                        messages=self.messages,
                        temperature=0.7,
                        max_tokens=500
                        )
            response_chunks = ""
            with result as stream:
                for text in stream.text_stream:
                    if usingGradio:
                        response_chunks += text or ""
                        yield response_chunks
                    else: 
                        yield text
        else:
            toSend = self.messages
            toSend.insert(0,self.system)
            stream = self.processor.chat.completions.create(
                model=self.model,
                messages= toSend,
                stream=True
            )
            response_chunks = ""
            for chunk in stream:
                if usingGradio:
                    response_chunks += chunk.choices[0].delta.content or "" # need to yield the total cumulative results to gradio and not chunk by chunk
                    yield response_chunks
                else:
                    yield chunk.choices[0].delta.content
