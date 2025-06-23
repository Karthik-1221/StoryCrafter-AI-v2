#  StoryCrafter AI

StoryCrafter AI is a creative web application that transforms your ideas or images into emotionally rich stories. It leverages ensemble NLP models to generate, score, and refine narratives across dramatic, humorous, or poetic tones.

---

##  Features

- **Image Captioning**: Extracts meaningful descriptions from uploaded images using BLIP.
- **Multi-Model Story Generation**: Generates stories using GPT-2, GPT-2 Medium, and GPT-Neo.
- **Bagging (Scoring & Selection)**: Evaluates stories for tone, readability, and uniqueness to select the best draft.
- **Boosting (Refinement)**: Enhances the top story using Flan-T5 Base for clarity and structure.
- **Text-to-Speech**: Converts stories into audio with gTTS.
- **Download Output**: Save your favorite story as a `.txt` file.

---

##  Run Locally

### 1. Clone the repository:
```bash
git clone https://github.com/Karthik-1221/StoryCrafter-AI-v2.git
cd StoryCrafter-AI
```

2. Install dependencies:
   pip install -r requirements.txt

3. Launch the app:
   streamlit run app.py
Tech Stack
- Python 3
- Streamlit (frontend)
- Hugging Face Transformers (GPT-2, GPT-Neo, Flan-T5, BLIP)
- gTTS (Text-to-Speech)
- scikit-learn, NumPy (Story scoring)
- textstat (Readability analysis)

Requirements
All dependencies are listed in requirements.txt. To install:
pip install -r requirements.txt
Recommended system: 64-bit machine with 8GB RAM
Optimized for CPU — no GPU needed.
 
Demo Workflow
- Upload an image (optional) and enter a prompt.
- Choose a tone: Dramatic, Humorous, or Poetic.
- Generate stories from all 3 models.
- Click "Apply Bagging" to score and select the best.
- Click "Apply Boosting" to refine the story.
- Listen, edit, or download your favorite version.





   
