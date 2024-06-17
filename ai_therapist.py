import getpass
import os

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

import requests
import json

import gradio as gr


class ExtendedConversationBufferMemory(ConversationBufferMemory):
    extra_variables:list[str] = []

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key] + self.extra_variables

    def load_memory_variables(self, inputs: dict[str, any]) -> dict[str, any]:
        d = super().load_memory_variables(inputs)
        d.update({k:inputs.get(k) for k in self.extra_variables})        
        return d


# del os.environ['NVIDIA_API_KEY']  ## delete key and reset
if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
    nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key

chat = ChatNVIDIA(
            model="meta/llama3-70b-instruct",
            temperature=0.1,
            max_tokens=4090,
            top_p=1.0,
        )    
    
template = """
You are a helpful AI therapist. 
Read the converstaion below, and give client the answer properly.
{context}

#conversation
{history}
----
Client: {input}
AI therapist:"""

PROMPT = PromptTemplate(input_variables=["history", "input", "context"], template=template)

conversation = ConversationChain(
    prompt=PROMPT,
    llm=chat,
    verbose=False,
    memory=ExtendedConversationBufferMemory(extra_variables=["context"])
)    

    
def summurize(conversation):
    summurizer = ChatNVIDIA(
        model="meta/llama3-70b-instruct",
        temperature=0.1,
        max_tokens=4090,
        top_p=1.0,
    )

    summary_prompt = PromptTemplate.from_template("summarize below {history} in 3 sentenses")
    summary_chain = summary_prompt | summurizer | StrOutputParser()

    input = {"history": str(conversation.memory.load_memory_variables({})['history'])}

    summary = summary_chain.invoke(input)
    
    return summary


def img2txt(file_path, summary):
    vision_language_model = ChatNVIDIA(model="nvidia/neva-22b")
    image_url = f"http://localhost:8000{file_path}"
    image_content = requests.get(image_url).content
    
    text = f"""{summary}
    Describe this image:"""

    image_description = vision_language_model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            )
        ]
    )
    
    image_description = image_description.content
    
    return image_description


with open("contexts.json") as file:
    contexts = json.load(file)

count = 0
file_path = ""
def response(message, history):
    global count
    global file_path
    global contexts
    global conversation
    
    if len(message["text"]) == 0 and count != 3:
        answer = "Empty String, please enter a valid text input."
    elif count == 3:
        if len(message["files"]) == 0 and len(message["text"]) != 0:
            answer = "Please upload an image and do NOT type anything."
        elif len(message["files"]) == 0:
            answer = "Please upload an image."
            answer = "Please do NOT type anything."
        else:
            answer = conversation.predict(
                input="",
                context=contexts[count]
            )
            count += 1
            file_path = message["files"][0]
    elif count == 4:
        summary = summurize(conversation)
        image_description = img2txt(file_path, summary)
        
        answer = conversation.predict(
            input="",
            context=f"Image description: {image_description}. {contexts[count]}"
        )
        count += 1
    else:
        if count >= len(contexts):
            answer = "It was nice meeting you! Our session has ended, but I look forward to seeing you next time."
        else:
            answer = conversation.predict(
                input=message["text"],
                context=contexts[count]
            )
            count += 1
            
    return answer


demo = gr.ChatInterface(
    response,
    chatbot=gr.Chatbot(placeholder="<strong>Hello, welcome. I am your AI Therapist.</strong><br> Before we start, can I get some information from you? (your name, age, and job)",
                       height=700),
    multimodal=True
)


if __name__ == "__main__":
    demo.launch(share=True)