# RAG optimization framework

A framework consisting of a lightweight Retrieval-Augmented Generation (RAG) system and a flexible benchmark infrastructure for systematic, use-case-specific optimization of RAG systems for individual knowledge management scenarios.


**This code supplements my master's thesis, titled:**

:de: "Intelligentes Wissensmamagement in KMU: Konzeption und Evaluation eines lokalen, RAG-basierten Wissensmanagementsystems"

:uk: "Intelligent Knowledge Management in SMEs: Design and Evaluation of a Local, RAG-Based Knowledge Management System"

:bulb: Abstract
> Small and medium-sized enterprises (SMEs) engaged in custom development accumulate substantial domain-specific knowledge that is frequently distributed across heterogeneous internal systems, limiting its accessibility and interpretability. Retrieval-Augmented Generation (RAG) presents a viable approach to address this challenge by enabling structured retrieval of fragmented information and facilitating interactive knowledge exploration. This thesis investigates the critical performance factors of RAG systems designed for knowledge management in resource-constrained SME environments with strict data sovereignty requirements, and proposes methods for their systematic quantification and optimization. A modular framework comprising a baseline RAG system and a flexible benchmarking infrastructure was developed to support use-case-specific, differentiated evaluation. An empirical optimization study conducted within this framework demonstrated statistically significant overall performance gains, with notable improvements in trustworthiness, response groundedness and context precision. The findings confirm that RAG systems can deliver immediate utility under simple configurations using publicly available components, while also establishing that no universally optimal configuration exists; optimization strategies must be evaluated contextually, as their efficacy is not transferable across arbitrary application scenarios or knowledge bases.

---

### What this repository contains
- the RAG module in it's final configuration after the optimization study in `/rag`
- the customizable benchmark infrastructure in `/benchmark`
- the notebooks for the evaluation of the benchmark results in `/eval`

### What this repository does not contain
- the benchmark results of from the thesis (due to them containing internal company data)
