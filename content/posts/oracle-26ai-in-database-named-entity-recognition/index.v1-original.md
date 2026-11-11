---
title: "Named Entity Recognition Inside Oracle AI Database 26ai: No External AI API Required"
date: 2026-11-11
draft: true
author: Hicham Assoudi
description: Running a custom Arabic BERT NER model inside Oracle AI Database 26ai on CPU, with entities returned to SQL and no external inference service.
summary: The text was already in Oracle, so why send it elsewhere to find the entities it contains? This overview covers the problem, the experiment I ran on Oracle AI Database 26ai, and the in-database NER solution it led to.
slug: oracle-26ai-in-database-named-entity-recognition
series:
  - Oracle 26ai AI Builder Series
categories:
  - Oracle Database
  - Artificial Intelligence
tags:
  - oracle ai
  - oracle database 26ai
  - implementation-guide
  - Oracle AI Database
  - ONNX
  - Named Entity Recognition
  - NLP
  - Arabic NLP
  - In-database AI
  - BERT
  - modern-nlp-on-oracle
cover:
  image: /images/in-db-ner-26ai-architecture.png
  alt: "Architecture diagram: a Hugging Face NER model is packaged once into ONNX outside the database, then runs inside Oracle AI Database 26ai on text columns and returns entity rows to applications, search, compliance, and analytics consumers"
  caption: The model is prepared once outside Oracle; inference and entity decoding run inside the database.
  relative: false
  hiddenInSingle: true
ShowToc: true
TocOpen: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowCodeCopyButtons: true
---

Enterprise databases contain more than structured business records. They also hold customer comments, incident descriptions, contracts, correspondence, and reports, often stored as text that applications can retrieve but cannot easily analyze.

Consider an Oracle database containing thousands of Arabic customer-service notes. A business user wants to know which people, organizations, and locations are mentioned across those records.

This is a **Named Entity Recognition (NER)** problem. In a conventional AI architecture, solving it means reading text from the database, sending it to an external NLP inference service, and writing the results back.

That approach works. But it introduces another service to deploy, integrate, secure, and operate, and it moves the source text across an additional processing boundary.

**What if the entire extraction could happen inside Oracle instead?**

This article is the overview: the problem, the experiment I ran on Oracle AI Database 26ai, and the solution it led to. The detailed test results and the architecture trade-offs follow in the next two articles of the series.

## The problem: the text is in Oracle, the NER is somewhere else

Suppose an application stores the following sentence in an Oracle text column:

> التقى أحمد العلوي بممثلي بنك المغرب في الرباط.

An NER application should recognize mentions such as these:

| Mention | Entity type |
| --- | --- |
| أحمد العلوي | Person |
| بنك المغرب | Organization |
| الرباط | Geopolitical location |

*The desired result. As the solution section shows, it is exactly what the in-database model returned for this sentence.*

Identifying these spans is different from classifying the entire sentence. A document classifier returns a single category. An NER model assigns a prediction to every token, and those predictions must be reconstructed into complete entities with exact boundaries.

The usual answer is an external inference service: a Python application or an API that reads the text, runs the model, and returns the entities. It works, at three costs:

- **Another component** to deploy, secure, monitor, and keep compatible with the database.
- **Another boundary** that the source text crosses, with its own access controls and audit trail.
- **Another synchronization problem**: results must be written back and kept consistent with the records they describe.

### What Oracle 26ai offers today

Oracle AI Database 26ai can import ONNX models as database `MINING MODEL` objects and run them through Oracle Machine Learning scoring operators. Oracle documents ONNX support for **classification, regression, clustering, and embeddings**.

None of those functions is token-level NER. There is no ready-made operator that returns extracted mentions, entity labels, and character offsets from a custom neural model.

Oracle does provide other entity-extraction options. Oracle Text's `CTX_ENTITY` uses dictionaries and rules, with built-in rules focused on English and facilities for customization. OCI Language provides managed NLP services. And Oracle's 26ai announcement lists named entity recognizers among the models its Private AI Services Container can run. That container is a separate service that the database calls over REST: it keeps data inside your own infrastructure, but it is still another component to deploy and maintain, and the text still leaves the database to reach it.

My question was narrower: **can a custom neural NER model run inside the database itself, called from SQL like any other function?**

## The experiment

The hypothesis was simple to state: if an NER model can be packaged in a form that 26ai's in-database ONNX runtime accepts, and its token-level predictions can be turned back into entities inside the database, then NER needs no external service at all.

### The model

I used [MagBERT-NER-AR](https://huggingface.co/TypicaAI/MagBERT-NER-AR), an Arabic NER model I developed through Typica.ai.

The model was **fine-tuned from an Arabic BERT backbone** on my own manually curated and annotated dataset drawn from Moroccan Arabic-language news. It is designed for **Modern Standard Arabic**, with particular attention to the entities and naming conventions found in that corpus.

It recognizes **12 entity types**: people, organizations, geopolitical entities, other locations, nationalities and groups, dates, events, laws, products, numbers, ordinals, and percentages. It uses a token-labeling scheme that makes entity reconstruction part of the inference workflow.

The training data matters here. For a business application, a model's usefulness depends not only on its architecture but on whether its training examples reflect the names, institutions, and terminology that users actually encounter.

Arabic was my main test case because I had both the model and the NLP experience to evaluate it. To check that the approach is not tied to one language or one tokenizer, I ran the same pipeline with a second model of mine, the French [magbert-ner](https://huggingface.co/TypicaAI/magbert-ner), which is based on CamemBERT and uses a different tokenization scheme.

### The setup

- **Database:** Oracle AI Database 26ai Free, running in Docker on a laptop, CPU only. Free is capped at 2 CPU threads and 2 GB of memory, a deliberately modest environment.
- **Test data:** 120 Arabic news-style sentences built around Moroccan people, institutions, places, laws, and events, covering all 12 entity types, plus 120 French sentences built the same way.
- **Workload:** the sentences were loaded into a table. A single SQL statement extracted the entities from every row and stored them in an entity table. Each phase was timed separately: loading the model, running it, and turning its output into entities.
- **Faithfulness:** during development, the database results were checked against the original model's predictions, to make sure that moving the model into Oracle did not change what it predicts.

## The solution

![High-level architecture of in-database NER on Oracle AI Database 26ai](/images/in-db-ner-26ai-architecture.png)

The solution has three parts:

1. **One-off preparation, outside the database.** The Hugging Face model is packaged into a single ONNX model that the in-database runtime accepts. This is the step that makes token-level output possible.
2. **Inference inside the database.** The packaged model is loaded once as a database model and runs on the database's CPUs, in Oracle's in-database ONNX Runtime.
3. **Entities back to SQL.** The model's token predictions are decoded inside the database into rows: entity type, exact text, character span, and confidence score. A single SQL call returns them, for one text or for a whole table.

Here is the sentence from the beginning of this article, run with one SQL call:

![One SQL call on Oracle AI Database 26ai returns three entities for the sentence التقى أحمد العلوي بممثلي بنك المغرب في الرباط: PERSON أحمد العلوي (characters 7 to 17, score 0.926), ORG بنك المغرب (26 to 35, score 0.712) and GPE الرباط (40 to 45, score 0.968)](/images/in-db-ner-query-problem-sentence.png)

*The three expected entities, each with its character span and confidence score, fetched in 0.078 seconds in a warm session.*

And a sentence from the test set:

![One SQL call returns two entities for the sentence اقتنت الخطوط الملكية المغربية ثلاث طائرات من طراز بوينغ 787: ORG الخطوط الملكية المغربية (characters 7 to 29, score 0.871) and PRODUCT بوينغ 787 (51 to 59, score 0.538)](/images/in-db-ner-query-test-sentence.png)

*The airline and the aircraft model are found. The number ثلاث is not tagged: gaps like this one are what the next article's quality evaluation looks at.*

In this implementation:

- **Inference runs on CPU.** No GPU is involved.
- **The source text stays inside the database for NER inference.** There is no external NER endpoint or API in the processing path.
- **Entities are plain rows**, available to SQL, joinable with the business records they came from, and produced inside the database's own transactions, access controls, and auditing.

I am keeping the model packaging and the decoding implementation private for now. The purpose of this article is to explain the engineering problem, what I have demonstrated, and the business applications it can support, not to present a reproducible deployment package.

## What the experiment showed

All 120 sentences per language were processed by one SQL statement in a single database session:

| Measure (Oracle AI Database 26ai Free, 2 CPU threads) | Arabic (MagBERT-NER-AR) | French (magbert-ner) |
| --- | --- | --- |
| First call in a session (model load) | 7.6 s | 7.5 s |
| Model inference | 45.7 ms per sentence | 54.2 ms per sentence |
| Decoding into entities | 1.5 ms per sentence | 1.2 ms per sentence |
| End-to-end, one SQL statement over the table | 46.9 ms per sentence | 55.1 ms per sentence |
| Throughput, one session | about 21 sentences per second | about 18 sentences per second |
| Entities extracted | 339 | 306 |

Three things stand out:

- **The model dominates the cost.** Turning predictions into entities inside the database adds about 1.5 ms per sentence, roughly 3% of the total.
- **The first call is expensive, the next ones are not.** Each session loads the model once, in about 7.5 seconds. Long-lived sessions or a connection pool absorb that cost.
- **The second language behaved the same way.** The French model, with a different tokenizer, went through the same pipeline with similar timings. Its sentences were longer on average (87 characters against 71).

These figures describe one session on a laptop running the Free edition. They are not production throughput: concurrency, memory per session, and the effect on other database workloads still need to be measured.

I also scored the extracted entities against the annotations of the synthetic test set. I am not publishing those scores here. Scoring against synthetic annotations mixes model quality with differences in annotation conventions (for example, whether the entity is «مدينة مراكش» or just «مراكش»), so the numbers need an error analysis before they mean anything. That analysis, on more representative text, is the subject of the next article.

## What this makes possible

NER turns passages of text into structured references to people, organizations, places, dates, and other domain-relevant entities. Once those annotations sit next to the records they came from, several applications open up:

- **Finding sensitive information in free-text records.** Structured personal-data columns are easy to identify; a customer-service note may contain a name or a location without its column saying so. NER supports discovery and review of those records. A complete PII solution also needs detectors for account numbers, identification numbers, phone numbers, and email addresses, plus policy controls and validation.
- **Searching documents by the entities they contain.** A legal team wants every document mentioning a given organization; a service team, every record involving a location or a customer. Persisted entities make those questions structured queries instead of keyword guesses.
- **Enriching AI search and RAG.** Vector search retrieves semantically similar text, but applications often need explicit filters: documents naming a specific company, person, or place. NER adds that metadata alongside keyword and vector retrieval.
- **Supporting document-intensive processes.** Contracts, claims, administrative records, and incident reports all contain information that workflows need to identify, route, and review. More specialized decisions still require additional models, rules, and human oversight.
- **Starting enterprise knowledge graphs.** Repeated mentions of people, organizations, and places are the raw material of a knowledge graph. Entity resolution and relationship extraction come next, but NER provides the initial structure.

These use cases share one property: **the organization already owns the text**. The new value comes from making that text queryable and operationally useful.

## Is an external inference service ever the better choice?

Yes. An external service is not inherently the wrong architecture. Dedicated inference infrastructure, including Oracle's Private AI Services Container, provides isolation and independent scaling, which matter for very large models, sustained high-throughput workloads, or applications that do not run on Oracle.

The point of this experiment is that such infrastructure is **not always necessary**. If an organization already runs Oracle AI Database 26ai and its text is stored there, in-database NER is an alternative worth evaluating. For suitable workloads it removes an integration component and keeps the source text where it already is.

That is an architectural benefit, not proof of lower total cost or better performance. Those claims require workload-specific measurement. The third article in this series compares the options in detail.

As a first filter:

| Your situation | In-database NER? | Why |
| --- | --- | --- |
| Text already stored in Oracle AI Database 26ai, compatible NER model available | ✅ Yes | No extra inference component; entities land next to the records they describe |
| Known terms in mostly English text, where dictionaries and rules are enough | ⚠️ Compare first | Oracle Text `CTX_ENTITY` may be sufficient without a neural model |
| Domain text far from the model's training data (contracts, customer notes) | ⚠️ Partial | Validate quality on representative text; domain fine-tuning may be needed |
| Sensitive-data discovery as part of a compliance program | ⚠️ Partial | NER finds names and places; account numbers, IDs, and policy controls need other detectors |
| Very large models or sustained high-throughput inference | ❌ No | Dedicated infrastructure, such as the Private AI Services Container, scales independently |
| Applications and text that do not live in Oracle | ❌ No | There is nothing to colocate with; use an external service |

## What I will validate next

The experiment shows that the approach works and gives a first measure of its cost on modest hardware. Before an enterprise deployment, I would want evidence in four areas:

1. **Extraction quality:** entity-level precision, recall, F1, and span boundaries on representative business text, with an error analysis.
2. **Execution under load:** latency, throughput, and memory with concurrent sessions, on production-grade hardware.
3. **Domain transfer:** how the news-trained model performs on customer notes, contracts, or other enterprise text, and whether domain-specific fine-tuning is needed.
4. **Operational controls:** model versioning, scheduling, permissions, auditing, and the effect of inference on other database workloads.

## Closing thought

The question that started this experiment was simple: **if the text is already in Oracle, why move it to another service just to identify the entities it contains?**

Using my own Arabic BERT-based NER model, I demonstrated that this extraction can happen inside Oracle AI Database 26ai, on CPU, called from SQL, with no external NER inference endpoint. On the Free edition, a single session processed about 21 Arabic sentences per second.

The opportunity is not limited to Arabic. It is a reusable way to bring specialized entity extraction to existing enterprise data, subject to model compatibility and workload validation.

**Next in this series:** how I tested in-database NER, quality and speed in detail. Then: where NER should run in an Oracle architecture, inside the database or next to it.

**If you are evaluating an Oracle-based document intelligence or sensitive-data discovery use case, I would be interested in discussing a focused proof of value using representative, appropriately controlled data.**

---

### References

- [Oracle AI Database 26ai: ONNX Support](https://docs.oracle.com/en/database/oracle/oracle-database/26/nfcoa/onnx-support.html)
- [Oracle Machine Learning: Integration of ONNX Runtime](https://docs.oracle.com/en/database/oracle/machine-learning/oml4sql/23/dmapi/onnx-runtime1.html)
- [Oracle Text: About Entity Extraction and CTX_ENTITY](https://docs.oracle.com/en/database/oracle/oracle-database/26/ccapp/entity-extraction-and-ctx_entity.html)
- [Oracle AI Database 26ai announcement (press release, 14 October 2025)](https://www.oracle.com/news/announcement/ai-world-database-26ai-powers-the-ai-for-data-revolution-2025-10-14/)
- [Oracle Private AI Services Container User's Guide](https://docs.oracle.com/cd/G47991_01/prvai/private-ai-services-container-users-guide.pdf)
- [MagBERT-NER-AR: model card on Hugging Face](https://huggingface.co/TypicaAI/MagBERT-NER-AR)
- [magbert-ner: model card on Hugging Face](https://huggingface.co/TypicaAI/magbert-ner)

*Hicham Assoudi, PhD: Computer Science Professor, AI/NLP Researcher, Oracle ACE Pro, and Founder of Typica.ai.*
