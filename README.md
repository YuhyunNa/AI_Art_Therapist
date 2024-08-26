# AI_Art_Therapist

### This repo is for Generative AI Agents Developer Contest by NVIDIA and LangChain
AI_Art_Therapist is designed to provide emotional healing by combining the power of AI technology with the principles of art therapy. I’ve obtained a certificate of accomplishment and been selected as a winner in the “Generative AI Agents Developer Contest” by NVIDIA and LangChain! Out of many talented developers around 31 countries, my “AI_Art_Therapist” project was ranked between 14th and 113th place.

![NvidiaLangChainCertificate](https://github.com/user-attachments/assets/aabd3305-0bcf-42ea-b78e-efa25ddc3eba)

This repo is LLM-powered applications utilizing LangChain and NVIDIA APIs.

- `llama3-70b-instruct`: Powers complex conversations with superior contextual understanding, reasoning and text generation.
- `neva-22b`: Multi-modal vision-language model that understands text/images and generates informative responses.

If anything stresses you out, try talking to AI Therapist to heal your emotions.
AI_Art_Therapist lets you draw a picture and AI_Art_Therapist will analyze the picture.


# Demo
- Click the image below to play the video.
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
