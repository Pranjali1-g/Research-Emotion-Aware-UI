# CARE-RAG Research Demonstrator

A polished, interactive Streamlit presentation app based on the paper:

**Emotion, Intent, and Risk-Aware Retrieval-Grounded Conversational Support for Cancer-Related Queries: A Matched-Comparison and Human-Evaluated Study**

Authors: Ikram Ali, Mannat Mehta, Pranjali Gupta, Yugam Rana  
Apex Institute of Technology, Chandigarh University

## Important implementation boundary

This repository is a **presentation-ready demonstrator**, not a reproduction of the trained experimental stack. The UI reproduces selected metrics reported in the supplied paper, while the live demo uses transparent keyword heuristics, TF-IDF retrieval over eight illustrative notes, and a deterministic response template. It does not include the paper's trained DistilBERT/DAPT weights, the 18-document research collection, or the Llama-3-8B-Instruct adapter. Do not describe the live demo as the paper's trained model.

This is an academic prototype only. It is not a medical device, diagnostic tool, treatment recommender, or emergency service. Do not enter real patient data.

## Features

- Responsive dashboard with paper summary and reported headline metrics
- Interactive fictional-query demo showing risk route, intent, emotion cue, retrieved notes, and response
- Risk-first architecture walkthrough
- Interactive Plotly charts and tables for emotion, intent/risk, retrieval, and human-evaluation results
- Study limitations and recommended next steps
- Research validation lab for user-supplied classification, retrieval, and paired human-rating CSVs
- Local TF-IDF + Logistic Regression and LinearSVC baselines with macro-F1, precision/recall, confusion matrix, and exportable held-out predictions
- Recall@K comparison for raw TF-IDF vs intent-guided TF-IDF; paired-rating analysis with exploratory significance tests
- Lightweight tests for routing behaviour

## Run on Windows

1. Install Python 3.10 or newer from https://www.python.org/downloads/
2. Extract this ZIP to a folder.
3. Open Command Prompt in that folder.
4. Run:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

5. Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.
6. For your presentation, use the left navigation and start with **Overview**, then **System architecture**, **Results & evaluation**, and finish with **Interactive demo**.

You can also double-click `run_windows.bat` after Python is installed.

## Project structure

```text
cancer_support_ui/
├── app.py
├── core.py
├── requirements.txt
├── README.md
├── run_windows.bat
├── data/
│   └── demo_evidence.json
├── tests/
│   └── test_core.py
└── .streamlit/
    └── config.toml
```

## Test the paper findings with your own data

Use the **Research validation lab** page after launching the app. It provides downloadable CSV templates and three experiment types:

1. **Classification:** upload rows with `text,label`. Select the task (emotion, intent paraphrases, adversarial risk, or clinician edge cases). The app makes a stratified hold-out split and compares TF-IDF + Logistic Regression with TF-IDF + LinearSVC. It exports metrics and held-out predictions.
2. **Retrieval:** upload a query CSV with `query,relevant_doc_ids,intent` and a document CSV with `doc_id,title,text` (title is optional). Relevant IDs are semicolon-separated. It reports Recall@K and hit rate for raw versus intent-guided TF-IDF.
3. **Human ratings:** upload paired ratings with `scenario_id,metric,deterministic,llm_adapter`. It calculates means, paired differences, paired t-tests, Wilcoxon tests, and Holm-adjusted t-test p-values where calculable.

**Research validity warning:** these experiments only test your supplied data. The app does not include the paper's original datasets, trained DistilBERT/DAPT models, sentence-embedding retriever, Llama adapter, or original rater-level observations. The paper's published scores shown in the UI are reference values, not freshly reproduced measurements. For a faithful replication, obtain the exact benchmark splits, model checkpoints, document collection, query relevance labels, and paired ratings, then document the preprocessing and random seeds. Do not present results from the sample templates as scientific evidence.

## Replace the demo with the actual trained research models

Recommended integration seams:
- Replace `analyze_message()` in `core.py` with versioned emotion, intent, and risk model adapters.
- Replace `data/demo_evidence.json` with the exact reviewed corpus and metadata used by the study.
- Keep risk routing independent and fail closed when model outputs are missing or uncertain.
- Replace `build_demo_response()` with a deterministic constructor or a retrieval-constrained LLM adapter.
- Implement source-level claim verification, lexical checks, logging, and evaluation before considering any real-world use.

Do not simply connect an LLM and label it clinically safe. The paper explicitly states that clinical effectiveness, safety, and deployment readiness are not established.
