# 🧠 AI & NLP Projects Collection

This repository showcases **three applied AI/NLP projects** built using the **Hugging Face Transformers library**, **PyTorch**, and classical ML tools like **Scikit-learn**.  
The projects demonstrate how modern deep learning models can be applied in **text classification, speech recognition, and natural language generation**.

---

## 📌 Projects Included

1. **Sentiment Analysis** → Classify tweets into positive/negative.  
2. **Speech-to-Text with Wav2Vec2** → Convert audio into human-readable text.  
3. **Text Generation** → Generate human-like continuations for poetry and prompts.  

---

## 📂 Repository Structure

```
📁 AI_NLP_Projects/
│── 📁 Sentiment_Analysis/
│   └── Sentiment_Analysis.ipynb
│
│── 📁 Hugging_Face_Speech_to_text/
│   └── Hugging_Face_Speech_to_text.ipynb
│
│── 📁 Text_Generation/
│   └── Text_Generation.ipynb
│
└── README.md
```

---

## 🚀 Project Overviews

### 1️⃣ Sentiment Analysis
- **Goal**: Detect the **polarity** of tweets (Positive / Negative).  
- **Dataset**: Tweets dataset (cleaned & filtered).  
- **Model Used**: Hugging Face `pipeline("sentiment-analysis")`.  
- **Architecture**: Transformer-based (BERT-like model).  
- **Workflow**:
  1. Load & clean tweets dataset.  
  2. Drop neutral entries, keep binary labels.  
  3. Run Hugging Face Sentiment Classifier.  
  4. Evaluate with Accuracy, F1-score, Confusion Matrix, ROC-AUC.  

📊 **Results**:
- Balanced performance on binary classification.  
- Clear separation in ROC curve, showing robust predictions.  

🔗 **Reference Model**: [Hugging Face Sentiment Pipeline](https://huggingface.co/docs/transformers/en/task_summary#sentiment-analysis)

---

### 2️⃣ Speech-to-Text (Automatic Speech Recognition)
- **Goal**: Convert spoken audio into text transcription.  
- **Dataset**: Audio files loaded with Librosa.  
- **Model Used**: `facebook/wav2vec2-base-960h`.  
- **Architecture**:
  - Pretrained on **960 hours of English speech**.  
  - Uses **CTC (Connectionist Temporal Classification)** loss for sequence alignment.  
- **Workflow**:
  1. Load raw audio (`.wav`) using Librosa.  
  2. Resample to 16kHz (required by Wav2Vec2).  
  3. Convert audio → token IDs with **Wav2Vec2Tokenizer**.  
  4. Decode predictions into human-readable text.  

📜 **Example Output**:
- Input: 🎤 *“Hello, welcome to AI world”*  
- Output: *hello welcome to ai world*  

🔗 **Reference Model**: [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)

---

### 3️⃣ Text Generation
- **Goal**: Generate **human-like text continuations**.  
- **Dataset**: Collection of **Robert Frost poems**.  
- **Model Used**: Hugging Face `pipeline("text-generation")` (GPT-like).  
- **Workflow**:
  1. Load dataset of poems.  
  2. Split into lines → prepare as prompts.  
  3. Initialize Hugging Face text generation pipeline.  
  4. Generate poetic continuations for given lines.  
  5. Test with custom NLP prompts.  

📝 **Example Output**:
- Input Prompt: *"The woods are lovely, dark and deep,"*  
- Generated Output: *"but the journey goes on, the silence speaks of dreams untold."*  

🔗 **Reference Model**: [Hugging Face Text Generation](https://huggingface.co/docs/transformers/en/task_summary#text-generation)

---

## ⚙️ Tech Stack
- **Languages**: Python  
- **Libraries**:  
  - `transformers` (Hugging Face)  
  - `torch` (PyTorch)  
  - `scikit-learn`, `numpy`, `pandas`  
  - `librosa` (for audio processing)  
  - `matplotlib`, `seaborn` (for visualization)

---

## 📌 How to Run
1. Clone the repo:  
   ```bash
   git clone https://github.com/ozarakesh533/AI_NLP_Projects.git
   cd AI_NLP_Projects
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Open notebooks in Jupyter/Colab:  
   ```bash
   jupyter notebook
   ```

---

## 📖 Future Work
- ✅ Enhance sentiment analysis with fine-tuned BERT/RoBERTa.  
- 🌍 Add multilingual speech-to-text support (Indian & European languages).  
- ✍️ Experiment with larger text generation models (GPT-2, GPT-Neo, LLaMA).  
- 🎨 Integrate visualization dashboards for results.  
- 🔊 Deploy ASR & Sentiment Analysis as **Flask API / Web App**.  
