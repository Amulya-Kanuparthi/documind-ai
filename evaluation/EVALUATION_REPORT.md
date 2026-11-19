

\# DocuMind AI — Retrieval Evaluation Report



\## 1. Project Overview



DocuMind AI is an enterprise document intelligence platform

that uses Retrieval-Augmented Generation (RAG) to answer

questions using supporting evidence from enterprise documents.



The system combines:

\- FAISS semantic vector retrieval

\- BM25 keyword-based retrieval

\- Hybrid retrieval using weighted Reciprocal Rank Fusion (RRF)

\- Groq-hosted LLM answer generation

\- Source citations and evidence display

\- Streamlit user interface



\## 2. Evaluation Objective



The objective was to measure how accurately the retrieval

pipeline identifies relevant documents for user questions.



Three retrieval strategies were evaluated:



1\. FAISS

2\. BM25

3\. Hybrid (FAISS + BM25)



\## 3. Dataset and Methodology



The evaluation used 20 benchmark questions derived from

synthetic enterprise policy documents.



Each question was associated with an expected relevant

document.



Retrieval was evaluated at K = 1, 3, and 5.



Metrics included:



\- Recall@K: Whether a relevant document appeared within

&#x20; the top K retrieved results.

\- Precision@K: Proportion of retrieved results considered

&#x20; relevant.

\- MRR@K: Mean Reciprocal Rank of the first relevant result,

&#x20; truncated at K.



Because each benchmark question has one relevant document,

Recall@K is equivalent to Hit@K in this evaluation.



\## 4. Quantitative Results



| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |

|---|---:|---:|---:|---:|

| FAISS | 0.950 | 0.950 | 1.000 | 0.963 |

| BM25 | 0.450 | 0.800 | 0.800 | 0.617 |

| Hybrid | 0.900 | 0.900 | 0.950 | 0.912 |



\### Recall Comparison



!\[Recall Comparison](figures/recall\_comparison.png)



\### MRR Comparison



!\[MRR Comparison](figures/mrr\_comparison.png)



\## 5. Failure Analysis



FAISS produced one Hit@1 failure.



The question concerned approvals required before modifying

a production environment. The relevant change-management

document appeared at rank 4.



BM25 produced 11 Hit@1 failures, including several cases

where the relevant document did not appear in the top five.



Hybrid produced two Hit@1 failures, involving financial

authorization and procurement requirements.



The results suggest that semantic retrieval handled

paraphrased policy questions better than keyword retrieval

on this benchmark.



\## 6. Hybrid Weight Optimization



Five RRF weight configurations were tested.



| FAISS Weight | BM25 Weight | Recall@1 | Recall@5 | MRR@5 |

|---|---|---|---|---|

| 0.50 | 0.50 | 0.900 | 0.950 | 0.912 |

| 0.60 | 0.40 | 0.900 | 0.950 | 0.912 |

| 0.70 | 0.30 | 0.900 | 0.950 | 0.912 |

| 0.80 | 0.20 | 0.850 | 0.950 | 0.887 |

| 0.90 | 0.10 | 0.850 | 0.950 | 0.887 |



Increasing the semantic retrieval weight did not improve

Hybrid performance.



The original equal-weight configuration was retained.



\## 7. Key Findings



1\. FAISS achieved the highest retrieval performance.

2\. FAISS retrieved the expected document within the top

&#x20;  five results for every benchmark question.

3\. BM25 performed less effectively on paraphrased questions.

4\. Hybrid retrieval performed strongly but did not exceed

&#x20;  standalone FAISS.

5\. Weight tuning showed that increasing semantic weight

&#x20;  does not necessarily improve reciprocal rank fusion.



\## 8. Limitations



\- The benchmark contained only 20 questions.

\- Documents were synthetic rather than real enterprise records.

\- Each question had one labeled relevant document.

\- Retrieval metrics do not measure answer correctness.

\- Weight configurations were evaluated on the same benchmark,

&#x20; so results may not generalize to unseen questions.

\- Citation format validation does not establish that cited

&#x20; evidence actually supports each generated claim.



\## 9. Future Improvements



\- Expand the benchmark with unseen questions.

\- Add multiple relevant documents per question.

\- Evaluate answer faithfulness and citation correctness.

\- Test retrieval on larger document collections.

\- Investigate reranking and query expansion.

\- Measure retrieval latency and resource usage.



\## 10. Conclusion



FAISS was the strongest retrieval method on the current

benchmark, achieving Recall@1 of 0.950, Recall@5 of 1.000,

and MRR@5 of approximately 0.963.



The evaluation established a reproducible baseline for

future improvements to DocuMind AI.



