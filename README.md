# 📊 InsightForge: AI-Powered Business Intelligence Assistant

## 🔍 Overview

**InsightForge** is a modular, AI-powered business intelligence (BI) assistant designed to help organizations—especially small and medium-sized enterprises—extract actionable insights from structured sales data and unstructured knowledge documents.

Built using **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, and **Streamlit**, this tool enables natural language querying, visual analytics, and automated evaluation in a single interface.

---

## 🎯 Features

- 📁 **PDF Knowledge Base Creation** using LangChain + Chroma
- 🤖 **BI Assistant** with Mistral LLM (local via LlamaCpp)
- 📈 **Interactive Dashboard** for sales, product, regional, and customer insights
- 🧠 **Retrieval-Augmented Generation (RAG)** for grounded LLM answers
- ✅ **LLM Evaluation** using QAEvalChain
- 💬 **Conversational Memory** to retain context

---

## 🛠️ Tech Stack

| Component              | Tools / Frameworks                                                   |
|------------------------|----------------------------------------------------------------------|
| Language Model         | [Mistral 7B](https://mistral.ai) via `LlamaCpp`                      |
| Embeddings             | `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` |
| Vector Store           | `Chroma`                                                             |
| Document Loader        | `PyPDFLoader`, `RecursiveCharacterTextSplitter`                     |
| BI Dashboard & UI      | `Streamlit`, `pandas`, `matplotlib`, `seaborn`                      |
| RAG System             | `LangChain`, `RetrievalQA`, `ConversationBufferMemory`              |
| Evaluation             | `LangChain.evaluation.QAEvalChain`                                   |

---

## 📁 Project Structure

```
.
├── build_vectorstore.py          # Script to load PDFs, chunk, embed, and store in Chroma
├── Insightdashboard.py           # Streamlit app with dashboard, BI assistant, and evaluation
├── sales_data.csv                # Structured sales dataset
├── PDF_Folder/                   # Folder with knowledge PDFs
├── vectorstore/                  # Persisted Chroma vector store
└── models/                       # Local Mistral model (GGUF format)
```

---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/insightforge-bi-assistant.git
cd insightforge-bi-assistant
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

> ⚠️ Also ensure you have the `mistral-7b-instruct-v0.1.Q4_K_M.gguf` model in the `models/` directory.

### 3. Build Vectorstore
```bash
python build_vectorstore.py
```

### 4. Run the App
```bash
streamlit run Insightdashboard.py
```

---

## 📸 App Features

- **📈 Dashboard Tab:** Filter sales by region, gender, product; view visual trends.
- **🧠 QA Evaluation Tab:** Upload test cases (questions + reference answers) to evaluate the model.
- **📊 Stats Summary Tab:** Get basic descriptive stats and sample data.
- **🤖 BI Assistant Tab:** Ask business questions and receive insights + actionable recommendations.

---

## 🧪 Sample QA Evaluation CSV Format

```csv
question,reference
"What is the average customer satisfaction for Widget A?","Around 3.014 with low standard deviation."
"Which region had the highest sales?","The region with the highest total sales is West."
"What is the median sales amount?","The median sales is approximately 552 units."
```

---

## 📌 Future Enhancements

- PDF upload support in the UI
- Better grading rubric for evaluations
- Persistent user sessions and history
- Role-based authentication for multi-user access

---

## 🙌 Acknowledgments

Developed as part of the **Applied Generative AI Capstone** for the course *"Advanced Generative AI"*.

---

## 🧑‍💻 Author

**Abin Joseph**  
[LinkedIn](https://www.linkedin.com/in/abin-joseph-409226105/) | [GitHub](https://github.com/abin222)
