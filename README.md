# Lightweight RAG prototype for breast cancer screening （pro-mri） ⚡️

**Lightweight Retrieval‑Augmented Generation system for breast cancer screening retrieval and question answering, using a single published paper as original knowledge material.**

---

## 📘 Overview

This repository implements a **minimal working RAG** system designed for breast cancer screening-related queries. It leverages one of my **published paper** as the primary knowledge source, enabling factual, concise, and interpretable responses based on domain-specific research.

Core components:
- **Knowledge Base:** https://doi.org/10.1007/s00330-022-08863-8
- **Embedding:** gemini-embedding-exp-03-07
- **Retriever:** local embedding-based search with ChromaDB.
- **Generator:** lightweight LLM (gemini-2.5-flash-preview-05-20) that consumes retrieved context and formulates answers.
---

## 🚀 Features

- ✅ **Paper-based knowledge corpus**: Q&A grounded in one actual published research.
- ✅ **Compact & Modular**: minimal dependencies and clear architecture.
- ✅ **Retrieval with citations**: context‐aware response generation with traceable sources.
- ✅ **Interactive demo**: chat or CLI interface for querying the system.
---

## 📁 Repository Structure

/ paper.md # the published paper
/ paper_chunk.py # paragraph based context chunk
/ paper_embed.py # chunk embedding, store vectors and database query function
/ README.md # This file

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jingxp/breast_cancer_screening_RAG.git
cd breast_cancer_screening_RAG
```

### 2. Set Up Environment

```bash
uv pip install -r pyproject.toml
```

### 3. Ask Questions!
```bash
uv run paper_embed.py"
```
Which ask the default question "What is the latest research on breast cancer screening?" 

## 🔎 Example

Question:

"Which modality is better for breast cancer screening, mammography or MRI?"

### RAG Answer:

```
"Based on the context provided:

Breast MRI is presented as being **better** than mammography for breast cancer screening in terms of detection capabilities.

The text states:
*   "Breast MRI could not only detect more breast cancers than mammography, but also detect cancers at an earlier stage \[1]."
*   "Especially for women with extremely dense breast, supplemental MRI has the potential to reduce interval cancers\[2]."

However, the context also highlights that "cost-effectiveness is still the biggest obstacle for a wider application of this sensitive modality\[4].
```

### Gemini 2.5 Flash answer
```
When it comes to breast cancer screening, both mammography and MRI play crucial roles, but they are often used in different contexts and for different populations. It's not a matter of one being definitively "better" than the other for all women, but rather which modality is more appropriate based on individual risk factors.

Here's a breakdown:  .......

```
We get a pro-MRI rag !

## 🧪 Future Improvements

Add evaluation metrics (e.g. precision, groundedness)
Support different LLMs
Support multi-paper ingestion
Integrate with LangChain
