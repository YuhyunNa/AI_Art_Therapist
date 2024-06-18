# AI_Art_Therapist

### This repo is for Generative AI Agents Developer Contest by NVIDIA and LangChain

This repo is LLM-powered applications utilizing LangChain and NVIDIA APIs.

- `llama3-70b-instruct`: Powers complex conversations with superior contextual understanding, reasoning and text generation.
- `neva-22b`: Multi-modal vision-language model that understands text/images and generates informative responses.

If anything stresses you out, try talking to AI Therapist to heal your emotions.
AI_Art_Therapist lets you draw a picture and AI_Art_Therapist will analyze the picture.


# Demo
- click the image below to play the video.
[![Watch the video](https://github.com/YuhyunNa/AI_Art_Therapist/assets/82826442/ad3a4373-f861-4cfe-bc28-87ec53f02c0c)](https://youtu.be/MjL734oDiKY?si=pyHSDN92rEyTuqlB)


# Quick Start
```
conda create -n therapy python=3.12
conda activate therapy

pip install langchain
pip install --upgrade --quiet langchain-nvidia-ai-endpoints
pip install gradio
```

```
cd /
python -m http.server
```

```
git clone https://github.com/YuhyunNa/AI_Art_Therapist.git
cd AI_Art_Therapist
python ai_therapist.py 
```
