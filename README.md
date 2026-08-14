# 🏥 Smart Prescription Reader: Clinical NER Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

## 📌 Overview
The **Smart Prescription Reader** is a web-based application designed to extract critical medical entities (such as drug names, dosages, and medical conditions) from unstructured clinical text and prescriptions. 
Built with a focus on digital health and data processing, this tool leverages a fine-tuned BioBERT model to transform raw, unstructured medical notes into clean, structured, and analyzable data.

## 🚀 Key Features
*   **Advanced NLP:** Utilizes the `biomedical-ner-all` pipeline via Hugging Face Transformers for highly accurate Medical Named Entity Recognition (NER).
*   **Automated Data Structuring:** Parses extracted clinical entities and presents them in a clean tabular format using Pandas—perfect for downstream data analysis and integration into healthcare databases.
*   **Interactive UI:** Deployed on Streamlit Community Cloud for a seamless, real-time user experience.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Machine Learning & NLP:** PyTorch, Hugging Face `transformers`
*   **Data Manipulation:** `pandas`
*   **Frontend & Deployment:** Streamlit

## 🌐 Live Demo
The application is live and accessible online: 
👉 **[Insert Your Streamlit App URL Here]**

## 💻 Local Installation
To test and run this project locally, clone the repository and install the required dependencies:

```bash
git clone [https://github.com/lianafarsi/Smart-Prescription-Reader.git](https://github.com/lianafarsi/Smart-Prescription-Reader.git)
cd Smart-Prescription-Reader
pip install -r requirements.txt
streamlit run app.py
