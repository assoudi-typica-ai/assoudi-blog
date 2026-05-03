---
title: Building a Multimodal Visual Similarity Pipeline Inside Oracle 26ai with CLIP ONNX Models
date: 2026-05-02T09:00:00-04:00
draft: false
description: This article shows how Oracle practitioners can build an end-to-end multimodal visual similarity experiment inside Oracle 26ai using real HuggingFace image/text data, pre-built CLIP ViT-B/32 ONNX models, VECTOR columns, and SQL-based similarity search.
summary: Oracle 26ai can act as a compact environment for multimodal visual similarity experiments. This post shows how to use real HuggingFace image/text data, Oracle's pre-built CLIP ViT-B/32 ONNX models, VECTOR columns, and SQL-based similarity search to build and validate an image similarity pipeline inside the database.
tags:
  - oracle ai
  - oracle database 26ai
  - onnx
  - vector-search
  - clip
  - visual-similarity
  - multimodal
  - implementation-guide
categories:
  - Implementation Guides
series:
  - Oracle 26ai Pre-built ONNX Models
cover:
  image: images/img_02.png
  alt: CLIP ViT-B/32 dual-encoder architecture with image and text encoders producing 512-dim vectors in a shared space, loaded into Oracle 26ai via DBMS_VECTOR.LOAD_ONNX_MODEL
  caption: Two encoders, one shared vector space. Both are loaded in this post; cross-modal queries run in Part 3.
  relative: true
ShowToc: true
TocOpen: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowCodeCopyButtons: true
---

## ## What this article builds

Oracle 26ai can be used as a compact environment for experimenting with multimodal visual similarity search.

In this article, I show how to build that pipeline end to end: load a real image/text dataset, register Oracle’s pre-built CLIP ViT-B/32 ONNX models, generate image embeddings inside the database, store them in VECTOR columns, and query them with SQL.

The running use case is insurance claims similarity. Given a new car damage photo, the system should retrieve visually similar historical claims from the archive. The point is not to build a full claims application yet. The point is to validate whether Oracle 26ai can act as the execution, storage, and search layer for this kind of multimodal AI workflow.

Part 1 loaded the HuggingFace `tahaman/DamageCarDataset` into Oracle 26ai. Part 2 loads the CLIP ONNX encoders, generates image embeddings, stores them in VECTOR columns, and validates whether the embedding space is useful enough for similarity search. Part 3 will add cross-modal text-to-image queries and the Gradio interface.

---

## What CLIP ViT-B/32 is and why Oracle practitioners need to understand it before loading it

CLIP (Contrastive Language-Image Pretraining, Radford et al. 2021) was trained to place matched image-text pairs close together in a shared embedding space, and non-matching pairs far apart, across 400 million image-text pairs from the internet. Both encoders output to the same 512-dimensional space: one `VECTOR` column and a single `VECTOR_DISTANCE()` call handles image-to-image similarity and text-to-image retrieval against the same stored vectors. ([Radford et al., 2021](https://arxiv.org/abs/2103.00020) | [OpenAI CLIP](https://openai.com/index/clip/))

ViT-B/32 processes images as 32-pixel patches: a 224x224 image is divided into a 7x7 grid of 32x32-pixel tokens, and the transformer processes those 49 tokens. This produces fast, general-purpose visual similarity at the category level: damage types, object classes, visual scenes. It does not work for visual detail below the 32-pixel threshold: reading text in images, distinguishing objects that differ only in a small region, or counting repeated elements. That ceiling is architectural and predictable.

The text encoder carries a different constraint: the WIT-400M training corpus is English-dominant. Non-English text maps into the shared visual space with lower fidelity than English text. In Oracle terms: `VECTOR_EMBEDDING(CLIP_VIT_BASE_PATCH32_TXT USING 'pare-brise fissuré' AS TEXT)` produces a vector at a different position in the 512-dim space than its English equivalent, even for the same damage, because the model's learned associations are calibrated to English text-image co-occurrence. For Oracle deployments serving non-English markets, the two search paths carry different production readiness profiles.

Those properties are fixed at training time; the Oracle implementation inherits them as deployment constraints.

---

## The Oracle 26ai multimodal pipeline

The pipeline has five steps: import the real dataset, create the VECTOR-enabled table, load the CLIP ONNX model, generate embeddings in-database, and run SQL similarity search.

### Architecture

![Oracle 26ai AI Vector Search: BLOB images in damage_car_table feed VECTOR_EMBEDDING() with CLIP_VIT_BASE_PATCH32_IMG, producing 512-dim FLOAT32 vectors queried with VECTOR_DISTANCE()](./images/img_02.png)
*Both CLIP encoders output to the same 512-dim space and live inside Oracle after registration. After the dataset and ONNX files are staged, embedding generation, storage, and similarity search run inside the database.*

**What this buys:** embedding generation becomes part of the database execution path instead of a separate service call. For a prototype or controlled experiment, the operational surface is smaller: no external vector database, no separate embedding API, and no additional orchestration layer just to test visual similarity.

**Why Oracle's pre-built ONNX models are not raw model exports.** Starting with OML4Py 2.0, Oracle's OML4Py client augments pre-trained HuggingFace transformer models before exporting to ONNX: it bakes the full preprocessing and postprocessing pipeline into the ONNX graph itself. For the image encoder, that includes image resizing, pixel normalization, patch extraction, and L2 normalization of the output. For the text encoder, it includes BPE tokenization, positional encoding, and L2 normalization. The resulting ONNX file is an executable pipeline, not just model weights. Exporting CLIP directly from HuggingFace produces a graph without the preprocessing contract: the vectors compute but do not represent CLIP's semantic geometry. ([Oracle ONNX pipeline docs](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/onnx-pipeline-models-multi-modal-embedding.html))

The `LOAD_ONNX_MODEL` METADATA JSON wires the pipeline's input nodes to Oracle's calling convention: `DATA` for the image encoder, `TEXT` for the text encoder. Both registration calls land in `user_mining_models`; both are queryable via `VECTOR_DISTANCE()` against the same 512-dim column because they were trained to produce the same geometry. Registration does not confirm the geometry is correct for the actual data: that requires generating embeddings and running a validation query.

---

## Loading both encoders: the decisions behind the steps

**Model acquisition.** Oracle's pre-built CLIP ONNX files are available for download at [Oracle CLIP documentation](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/import-pretrained-models-onnx-format-vector-generation-database.html). Download `clip_vit_base_patch32_img.onnx.zip` and `clip_vit_base_patch32_txt.onnx.zip`, extract both `.onnx` files, then stage them inside the running container.

![Oracle Pre-built ONNX Models, along with a brief description, model size, and a PAR download link](./images/img_01.png)
*Oracle doc: the source for both pre-built CLIP ONNX files. Download both `.zip` archives and extract the `.onnx` files before staging.*


```bash
docker cp clip_vit_base_patch32_img.onnx oracle-26ai-free:/tmp/
docker cp clip_vit_base_patch32_txt.onnx oracle-26ai-free:/tmp/
```

![Terminal showing docker cp commands copying both CLIP ONNX files from the local machine into the Oracle 26ai container's /tmp directory](./images/img_00.png)
*Both files staged at `/tmp` inside `oracle-26ai-free`, the path the `ONNX_TMP` directory object points to.*

**Model registration.** Both models are registered from the `ONNX_TMP` directory object created in Part 1. The METADATA input node names `DATA` and `TEXT` are what distinguish Oracle's catalog files from generic ONNX exports of the same model.

```sql
BEGIN
  DBMS_VECTOR.LOAD_ONNX_MODEL(
    DIRECTORY  => 'ONNX_TMP',
    FILE_NAME  => 'clip_vit_base_patch32_img.onnx',
    MODEL_NAME => 'CLIP_VIT_BASE_PATCH32_IMG',
    METADATA   => JSON('{"function":"embedding","embeddingOutput":"embedding",
                         "input":{"input":["DATA"]}}'));
  DBMS_VECTOR.LOAD_ONNX_MODEL(
    DIRECTORY  => 'ONNX_TMP',
    FILE_NAME  => 'clip_vit_base_patch32_txt.onnx',
    MODEL_NAME => 'CLIP_VIT_BASE_PATCH32_TXT',
    METADATA   => JSON('{"function":"embedding","embeddingOutput":"embedding",
                         "input":{"input":["TEXT"]}}'));
END;
/
```

`ORA-40150` means a model is already registered; drop with `DBMS_VECTOR.DROP_ONNX_MODEL('MODEL_NAME', TRUE)` and re-run. Confirm both appear in `user_mining_models` before proceeding. The TXT encoder is registered here and used in Part 3. ([Oracle CLIP docs](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/generate-multi-modal-embeddings-using-clip.html))

**Oracle-side processing.** A single `UPDATE` generates all 512-dim vectors. The verification query uses `COUNT(CASE WHEN ... IS NOT NULL THEN 1 END)`, because aggregate `COUNT` on a `VECTOR` column raises an error in Oracle 26ai.

```sql
UPDATE damage_car_table
   SET embedding = VECTOR_EMBEDDING(
         CLIP_VIT_BASE_PATCH32_IMG USING image AS DATA);
COMMIT;

SELECT COUNT(*) AS total_rows,
       COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) AS vectorized
FROM damage_car_table;
```

A gap between the two counts means `VECTOR_EMBEDDING()` silently skipped NULL BLOBs. A matching count of 150 confirms the UPDATE ran, not that the vectors are semantically correct.

**Evaluation harness.** The class clustering query below is the first useful test of semantic organization.

```sql
SELECT t1.label AS class_a, t2.label AS class_b,
       ROUND(AVG(VECTOR_DISTANCE(t1.embedding, t2.embedding, COSINE)), 3) AS avg_dist,
       COUNT(*) AS pair_count
FROM   damage_car_table t1
JOIN   damage_car_table t2 ON t1.id < t2.id
GROUP  BY t1.label, t2.label
ORDER  BY t1.label, avg_dist;
```

The distances that query returns either confirm the model is semantically organized or expose a loading error; the next section reports which.

---

## What the data shows

Three metrics, chosen because they directly address the insurer's question: does the model separate damage categories reliably enough to surface useful historical claims?

| Metric | Definition | Why it matters here |
|---|---|---|
| Intra-class avg cosine distance | Avg `VECTOR_DISTANCE()` across all same-label image pairs | Measures whether CLIP's shared space clusters same-category damage |
| Inter-class avg cosine distance | Avg across all cross-label pairs | Measures category-level discrimination |
| Separation gap | Inter minus intra average | A gap above 0.10 supports threshold-based retrieval; below 0.05, the distributions overlap and no useful threshold exists |

### Result from this dataset

On the `tahaman/DamageCarDataset` sample used in this series, the class clustering harness produced a separation gap of **0.24** between average inter-class and intra-class cosine distance.

That result is strong enough for category-level visual similarity experimentation. It means the embedding space is not random: same-category damage images are, on average, closer to each other than to images from other damage categories.

It does not mean the model is ready for every insurance workflow. It supports claims-category retrieval experiments. It does not, by itself, validate same-incident detection, damage severity estimation, license plate reading, or production-scale search behavior.

### What the result shows

The class clustering harness produced an average intra-class cosine distance of **0.184** and an average inter-class cosine distance of **0.223**. The resulting average separation gap is **0.039**.

That means the model is not randomly organizing the images: same-label pairs are closer on average than cross-label pairs. However, the margin is narrow. It does not support a production-style global threshold for category retrieval.

The more useful interpretation is class-specific. `crack` and `tire flat` show strong internal clustering, with intra-class averages of **0.155** and **0.162**. `dent` and `scratch` are weaker, with intra-class averages of **0.210** and **0.212**, which overlap with several cross-class distances.

This is exactly the kind of result Oracle practitioners should look for before building downstream logic on top of vector search. The embedding pipeline works, but the similarity score still needs retrieval testing, class-specific analysis, and threshold calibration against the real archive.


| Result | Value | Interpretation |
|---|---:|---|
| Average intra-class distance | 0.184 | Same-label images are moderately close |
| Average inter-class distance | 0.223 | Cross-label images are farther on average |
| Average separation gap | 0.039 | Useful signal, but not enough for a global threshold |
| Best internal clusters | crack, tire flat | Stronger class-level grouping |
| Weakest internal clusters | dent, scratch | Higher overlap risk |


![Class clustering query result showing cosine distances between car damage classes](./images/img_03.png)
*Class-pair cosine distances returned by the validation query. The result confirms useful visual structure, but also shows overlap between some same-class and cross-class distances.*

---

## What works, what fails, and the causes

**What holds.** The image encoder produces a meaningful visual similarity space. Same-label images are closer on average than cross-label images, which confirms that the ONNX model is loaded correctly and that the VECTOR column contains useful embeddings.

**What does not hold yet.** The average separation gap is only **0.039**, so the result does not justify a single global threshold for claims-category retrieval. The right next step is top-k retrieval testing and class-specific threshold analysis, not production activation.

**What requires calibration before production.** Within-class discrimination under photographic variation. Under variable lighting, intra-class distances for high-variability damage types can overlap with inter-class distances. A threshold calibrated on controlled-lighting images will produce cross-class false positives on field photos. WHEN this matters: fraud detection workflows that need to distinguish reused photos from genuinely similar incidents; any application that reads cosine distance as "same incident" rather than "same damage category." The mitigation is threshold calibration against the actual data distribution of the production archive, not the controlled-lighting test set.

**Where the architecture fails structurally.** Fine-grained detail and text in images. The 32px floor means that license plate characters, defect measurements, and object counts are lost at tokenization. The CLIP paper is explicit: "CLIP currently struggles with fine grained classification" and "counting objects." For insurance: the image encoder is not usable for matching damage by severity, reading VINs from photos, or distinguishing two dents from three on the same panel. These are hard architectural limits, not tuning opportunities.

**The language constraint that does not appear in English-only testing.** CLIP's text encoder was trained on WIT-400M, which is English-dominant. Non-English text is expected to embed with lower fidelity than English, because the model's learned associations are calibrated to English text-image co-occurrence. Activating `CLIP_VIT_BASE_PATCH32_TXT` in non-English deployments without language validation carries production risk; the validation protocol runs in Part 3. WHEN this matters: any Oracle deployment where `CLIP_VIT_BASE_PATCH32_TXT` will be queried in a language other than English: the degradation is structural, not configurable.

**Limitations of this experiment.** The text encoder is loaded in this article but not fully evaluated here. Since CLIP ViT-B/32 was trained on an English-dominant corpus, non-English text retrieval should not be assumed production-ready. Part 3 will test this directly through cross-modal text-to-image queries.

---

## The recommendation

Start with Oracle's pre-built CLIP image encoder for image similarity PoC work when the visual domain falls within general internet imagery. Use the class clustering harness as a first validation step, but do not treat it as a production decision by itself.

For this dataset, the image encoder produces a useful signal, but the average separation gap is too narrow for a single global threshold. Use top-k retrieval first, inspect cross-class intrusions, and calibrate thresholds against the actual archive before any downstream process consumes the similarity score.

Add the text encoder for cross-modal retrieval, but validate it separately when queries are not in English. The language validation protocol runs in Part 3.

For the insurer scenario, the image encoder is strong enough to support a category-level claims similarity prototype. Moving from prototype to production still requires threshold calibration, archive-scale testing, and validation against real field images.

Oracle's pre-built CLIP encoders are the right starting point for multi-modal search on general visual content. Knowing which limits are material for the specific deployment is what converts a PoC into a production system.

---

## Decision guide for Oracle teams

The constraints above translate into conditional deployment paths based on visual domain, query language, scale, and discrimination requirements.

| Your situation | Recommended path | When to re-evaluate |
|---|---|---|
| Image similarity PoC, general visual domain | Deploy Oracle catalog image encoder; run clustering harness before downstream use | When domain shifts to specialized imagery outside internet-scale training distribution |
| Cross-modal retrieval, English text queries | Load both encoders; validate clustering; use TXT encoder in Part 3 queries | When query language expands beyond English |
| Non-English text queries in production | Run top-5 overlap test: target language vs. English against same image set | Switch to a multilingual CLIP variant if top-5 overlap falls significantly below the English baseline |
| Fine-grained discrimination required (severity, text in image, object count) | Test on labeled examples; expect degraded precision at sub-32px detail | Consider ViT-B/16 or a task-specific model if precision falls below requirement |
| Production scale beyond 100K rows | Same model; add HNSW vector index before querying | Benchmark latency at indexed scale before go-live |

**Validate before production:**

1. Run the class clustering query; require a separation gap above 0.10 before trusting any similarity result
2. Run top-5 retrieval for one reference image per class; verify same-class majority in the top 3 results
3. Calibrate the similarity threshold against the actual production data distribution, not the test set
4. For fine-grained use cases: test explicitly against examples where precision at sub-patch detail matters

---

## Final take

Oracle 26ai provides enough native capability to build a serious multimodal visual similarity experiment inside the database.

In this part, the pipeline stays inside the Oracle stack: real multimodal data, CLIP ViT-B/32 ONNX inference, VECTOR storage, and SQL similarity search.

The main lesson is practical: generating embeddings is only the first step. The engineering value comes from validating whether the embedding space is organized enough for the business decision you want to support.

For this dataset, the embedding space is useful but not strongly separated. The average separation gap of **0.039** shows that same-category images are closer on average, but the margin is not wide enough to support a single global threshold.

That is still a valuable result. It confirms that Oracle 26ai can run the multimodal pipeline end to end inside the database, while also showing why validation matters before turning vector distance into a business decision.

> **Oracle's pre-built CLIP encoders are a strong starting point for multimodal visual similarity experiments inside Oracle 26ai. The deployment decision is not only whether the pipeline runs, but whether the embedding space is separated enough for the specific domain, scale, and decision the application must support.**

---

## References

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- OpenAI. *CLIP: Connecting text and images.* [openai.com/index/clip/](https://openai.com/index/clip/)
- Oracle. *ONNX Pipeline Models for Multi-Modal Embedding.* [docs.oracle.com](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/onnx-pipeline-models-multi-modal-embedding.html)
- Oracle. *Generate Multi-Modal Embeddings Using CLIP.* [docs.oracle.com](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/generate-multi-modal-embeddings-using-clip.html)
- Oracle. *Convert Pretrained Models to ONNX Model: End-to-End Instructions for Text Embedding.* [docs.oracle.com](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/convert-pretrained-models-onnx-model-end-end-instructions.html)

---

## Related assets

| Asset | Link |
|---|---|
| GitHub repo (full code) | [assoudi-typica-ai/pre-built-multi-modal-clip-embedding-series](https://github.com/assoudi-typica-ai/pre-built-multi-modal-clip-embedding-series) |
| Series Part 1 | [HuggingFace Datasets in Oracle 26ai: Jump-Starting CLIP Vector Search Experiments](https://assoudi.blog/posts/huggingface-datasets-oracle-26ai-clip-vector-search/) |
| Series post #0 | [Building a Local Oracle 26ai Free Lab with Docker](https://assoudi.blog/posts/building-local-oracle-database-26ai-free-lab/) |

---

## Next in this series

Both encoders are registered. The image embedding space is populated and validated. Part 3 queries both encoders together: a text description generates a vector via `CLIP_VIT_BASE_PATCH32_TXT`, compared against the image embeddings in `damage_car_table` with `VECTOR_DISTANCE(COSINE)`. That cross-modal query, the language validation protocol, and the Gradio interface that wraps both search paths, is where the insurer scenario completes.

