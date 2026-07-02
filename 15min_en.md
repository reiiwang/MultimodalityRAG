# 15-Minute Paper Sharing: Five Multimodal RAG Papers

---

## Paper 1｜ColPali（2024）

### Current Problem

Existing document retrieval systems rely on **complex preprocessing pipelines** (OCR → layout detection → text chunking → embedding), which fundamentally compress visual documents into plain text, leading to:

- **Visual cues — charts, tables, page layout — are directly discarded**
- OCR and layout detection are themselves sources of error; one failure cascades through the whole pipeline
- Indexing is extremely slow (captioning strategies can take tens of seconds per page)

### Research Question

> **Can we skip text extraction entirely and embed document page images directly as vectors, retrieving by visual features?**

### Scope

Page-level document retrieval (PDFs, scientific papers, slides, and other visually rich documents), balancing three industrial requirements: retrieval accuracy, query latency, and indexing throughput.

### Method & Highlights

**Core Architecture: ColPali = VLM (PaliGemma-3B) + ColBERT Late Interaction**

Traditional approaches need the entire preprocessing pipeline before an embedding model can even see the document. ColPali's core insight: **since VLMs already understand images, why not feed page screenshots directly?**

**Indexing (Offline)**

Each document page is fed as an image into PaliGemma-3B. The VLM splits the page into image patches, producing one token embedding per patch, all stored in the index. **Each vector corresponds to a local region of the page**, preserving spatial information.

**Query (Online)**

A text query is fed into the same VLM, producing a sequence of query token embeddings.

**Matching: Late Interaction**

Borrowed from ColBERT — instead of compressing the whole page into a single vector, each query token "scans" all patch embeddings in the document and finds the most similar one; scores are then summed across all query tokens:

$$LI(q,d) = \sum_{i} \max_{j} \langle E_q(i) \mid E_d(j) \rangle$$

This means every word in the query can align to the most relevant region of the page (e.g., "third column of the table," "bottom-left chart"), giving much finer-grained matching than single-vector retrieval, while remaining faster than a cross-encoder because **document embeddings are precomputed and cached**.

**Training**: End-to-end fine-tuning with in-batch contrastive loss — the correct page is the positive sample, other pages in the batch are negatives.

**Highlights**:
- Completely bypasses OCR, layout detection, and captioning — **indexing is 100× faster than captioning-based approaches**, with higher accuracy
- VLM pretraining has already aligned text and image patch representations, so fine-tuning cost is low
- Simultaneously releases **ViDoRe**, a page-level document retrieval benchmark covering medical, business, and scientific domains with charts, tables, infographics, and more

---

## Paper 2｜VLM2Vec-V2（2025）

### Current Problem

First, clarify the paper's goal: **VLM2Vec-V2 trains a universal multimodal Embedding Model**, not a generative QA model. It compresses diverse visual inputs into fixed-dimensional vectors so that semantically similar inputs are close in vector space, enabling downstream tasks like retrieval and classification.

Three core obstacles in existing methods:

1. **Modality heterogeneity**: Text is discrete sequences, images are pixel grids, video has a temporal dimension. Traditional approaches train separate encoders per modality, making cross-modal comparison and scaling difficult
2. **Task diversity**: The same "image + text" input requires different embeddings for image-text retrieval vs. image classification — the model has no way to know "what it should be doing right now"
3. **Insufficient modality coverage**: Existing models almost exclusively support static images, with very poor support for **video** (temporal information) and **visual documents** (PDF/slide layout structure)

### Research Question

> **How do we convert a generative VLM into a unified multimodal Embedding Model, letting it distinguish task intent through natural language instructions while supporting images, video, and visual documents?**

### Scope

Unified embedding across four input modalities — text, image, video, visual document — covering retrieval, classification, and QA task formats (78 datasets total).

### Method & Highlights

**Two contributions released together:**

**① MMEB-V2 Benchmark**

Extends the original MMEB (image + text) with five new task types:

| Task | Description |
|---|---|
| Video Retrieval (V-RET) | Text description → find the correct video from thousands of candidates |
| Moment Retrieval (M-RET) | Text → find the correct time segment in a video (~10 candidate segments) |
| Video Classification (V-CLS) | Video → predict scene/action category label |
| Video QA (V-QA) | Video + question → select the correct answer from multiple choices |
| Visual Document Retrieval (VisDoc) | Text query → find the correct PDF/slide page |

**② VLM2Vec-V2 Embedding Model**

**Backbone: Qwen2-VL — necessary but not sufficient**

Qwen2-VL solves the "input side" problem: Naive Dynamic Resolution handles variable-resolution inputs, M-RoPE captures spatial and temporal structure, and unified 2D/3D convolutions process images and video under one architecture.

But Qwen2-VL is a generative model (next-token prediction) — it does not automatically produce a vector space where "semantically similar inputs are close," and it has no notion of "am I doing retrieval or classification right now?" The backbone is just the foundation; the training methodology fills the gaps.

**Converting a generative model into an embedding model: last-token pooling**

After feeding input into Qwen2-VL, take the vector of the **last layer's last token** as the embedding for the entire input. Why the last token? Because Qwen2-VL uses causal attention — the last token has "attended to" all preceding tokens and theoretically encodes the full input semantics.

**Dual-Encoder Architecture**

Query and target run **separate, independent forward passes** — they are not concatenated:

```
[instruction + query content] → Qwen2-VL → last token → h_q
[target content]              → Qwen2-VL → last token → h_t
```

This allows targets to be **pre-encoded offline into a vector database**; at query time only a cosine similarity computation is needed, making retrieval extremely fast. (Concatenating them would create a cross-encoder, which must re-run for every new query and cannot be precomputed.)

**Instruction-Guided Contrastive Training (InfoNCE Loss)**

Contrastive training shapes the embedding space so that semantically similar inputs are close. A natural language instruction prepended to the query tells the model what task it is performing:

$$\mathcal{L} = -\log \frac{\varphi(h_q^{inst}, h_{t^+})}{\varphi(h_q^{inst}, h_{t^+}) + \sum_{t^- \in \mathcal{N}} \varphi(h_q^{inst}, h_{t^-})}, \quad \varphi(h_q, h_t) = \exp\!\left(\tfrac{1}{\tau}\cos(h_q, h_t)\right)$$

This is essentially a **multi-class classification problem** — softmax over all candidates, maximize the probability of the correct target:

- **In-batch negatives**: Other samples' targets in the same batch automatically become negatives — batch size B gives B−1 free negatives per query
- **Hard negatives**: Deliberately selected "confusable" samples that force the model to learn fine-grained semantic distinctions
- Gradients simultaneously pull correct pairs closer and push all negatives apart, sculpting a semantically meaningful embedding space

**Video and its challenges**: Videos are uniformly sampled into frame sequences fed as a whole, preserving temporal coverage. Moment retrieval (M-RET) is especially hard — the model must identify the correct time segment from ~10 candidates, requiring genuine temporal understanding rather than just static visual matching.

- Outperforms all prior baselines across 78 datasets
- A single model switches task behavior via instruction, no need to deploy separate models per task
- MMEB-V2 is itself a contribution — no prior benchmark simultaneously evaluated all four visual modalities

---

## Paper 3 & 4｜RAG-Anything ＋ MMGraphRAG（2025）

> Presented together because they address the exact same problem and each serves as the other's baseline.

### Current Problem

Traditional RAG/GraphRAG systems **only handle plain text**, but real-world documents (academic papers, financial reports, news, legal documents) are inherently **multimodal**, containing text, images, tables, and equations.

This creates shared core pain points:

- Existing systems either **discard non-text content** or **force-convert charts into incomplete text descriptions**, causing severe information loss
- Cannot answer questions that require looking at a figure or table (e.g., "which model performs better in the chart," "what is the value in a specific cell of the financial table")
- In **long documents** (100+ pages), relevant evidence is scattered across modalities and sections, making text-only retrieval ineffective

> Both papers choose the same route: building a knowledge graph that fuses text and visual information. They use **the same evaluation benchmarks** (DocBench, MMLongBench) and each appears as the other's baseline.

---

### Solution Concepts, Scope, and Methods

#### 3.1 RAG-Anything（HKU, 2025）

**Core concept**: Treat all modalities (text/image/table/equation) as "atomic knowledge units," represent them with a **dual-graph architecture**, then perform cross-modal hybrid retrieval.

**Scope**: Text, images, tables, equations — **all modalities**, the broadest coverage.

**Method (three stages)**:

**① Multimodal Knowledge Unification**

MinerU parser decomposes the document into atomic knowledge units $c_j = (t_j, x_j)$, where $t_j$ is the modality type (text/image/table/equation) and $x_j$ is the raw content. For each non-text unit, the system:
- Uses a VLM to generate two kinds of descriptions: a **detailed description** (for semantic understanding) and an **entity summary** (for graph extraction)
- Considers **contextual neighbors** (e.g., the caption paragraph next to a figure) to produce more accurate descriptions

**② Dual-Graph Construction — Core Innovation**

Two knowledge graphs are built simultaneously:
- **Cross-Modal KG**: Non-text units as nodes; entity relations extracted from VLM-generated descriptions; structural links preserved (e.g., `table_header → belongs_to → table_cell`, `chart_panel → described_by → caption`)
- **Text-based KG**: Traditional LightRAG/GraphRAG-style entity-relation extraction on plain text segments

The two graphs are merged into a unified graph $G$ via **entity alignment** (entity names as primary matching keys), with a vector embedding table $T$ built in parallel.

**③ Cross-Modal Hybrid Retrieval**

Retrieval runs two paths in parallel:
- **Structural navigation**: Multi-hop reasoning over the knowledge graph, best for evidence with explicit structural links (e.g., following `belongs_to` edges from a table cell back to the full table)
- **Semantic similarity matching**: Vector retrieval, best for semantically related content without direct structural edges

Results from both paths are merged and re-ranked with multi-signal fusion. Visual chunks undergo **dereferencing** (the original image is fetched back), and a VLM generates the final answer conditioned on both the "text context + original visual content."

**Strengths**: Innovation spans both the indexing and retrieval stages; particularly strong on **long documents** and **structurally complex tables/multi-panel figures** (case studies show correct localization of table row-column intersections and correct sub-figure identification in multi-panel images).

---

#### 3.2 MMGraphRAG（HIT & NTU, 2025）

**Core concept**: Convert images into **Scene Graphs**, then use a specially designed cross-modal entity linking method **SpecLink** to align image entities with text entities and fuse them into a unified MMKG.

**Scope**: Primarily **text + images** (tables and equations are still treated as plain text, with no special modeling).

**Method (core task: CMEL)**:

**① Image2Graph**

YOLO performs semantic segmentation on images, carving out visual regions (people, objects, scene elements). An MLLM generates text descriptions for each region, from which entities and relations are extracted. A key detail: extracted relations include not only explicit relations (e.g., "A is holding B") but also **implicit relations inferred by the MLLM** (e.g., "the two people are standing close together, possibly a couple"), enriching image semantics. A "global image entity" is also created to represent the whole image, forming a hierarchical structure with local regions.

**② Text2Graph**

Traditional GraphRAG-style entity and relation extraction on text paragraphs, building a text knowledge graph.

**③ Cross-Modal Fusion: SpecLink — Core Innovation**

The challenge: a text graph may have thousands of entities; brute-force pairwise comparison between every visual entity and every text entity is both inefficient and error-prone. SpecLink solves this in two steps:

- **Spectral clustering to narrow the candidate pool**: A redesigned weighted adjacency matrix encodes both "entity semantic similarity" and "graph neighborhood relation weights"; spectral clustering groups visual and text entities together so each visual entity only needs to search for matches within its cluster
- **LLM for final decision**: From the narrowed candidate set, an LLM makes the final alignment judgment

Compared to KMeans/DBSCAN (semantic only) or PageRank (graph structure only), SpecLink leverages both. Ablation experiments show that replacing SpecLink with a naive similarity-threshold approach drops accuracy on "unanswerable" questions by 9.7 percentage points.

**④ Retrieval and Generation**

Multi-hop retrieval along reasoning paths in the fused MMKG; answers generated by an LLM+MLLM hybrid (LLM for text portions, MLLM for visual portions).

**Additional contribution**: Because the CMEL (Cross-Modal Entity Linking) task lacked a standard benchmark, the authors also construct and release a **CMEL dataset** (news/academic/fiction domains) specifically for evaluating entity alignment quality.

**Strengths**: Innovation concentrated on the single but critical challenge of **correctly pairing image entities with text entities**; particularly strong at **identifying unanswerable questions** (hallucination suppression).

---

### Comparison Summary

| Dimension | RAG-Anything | MMGraphRAG |
|---|---|---|
| Modality coverage | Text + image + table + equation (all modalities) | Text + image (tables/equations still plain text) |
| Innovation focus | Dual-graph architecture + cross-modal hybrid retrieval (indexing & retrieval) | SpecLink cross-modal entity linking (spectral clustering), focused on graph fusion |
| Additional contribution | No new dataset | Releases CMEL dataset for evaluating cross-modal entity alignment |
| Relative strength | Long documents, complex tables / multi-panel figures | Hallucination suppression, identifying unanswerable questions |

> **One-sentence summary**: Both papers address the exact same problem (multimodal document RAG QA), but RAG-Anything takes a "**unified all-modality framework**" approach with innovations spread across indexing and retrieval; MMGraphRAG takes a "**precise text-image entity alignment**" approach, concentrating effort on SpecLink as a single technical breakthrough, validated with a new dataset.

---

### A Deeper Commonality: Both Follow the "Dual Graph → Fusion" Paradigm

```
Text data   → Text-based KG       ┐
                                   ├→ Fusion → Unified MMKG → Retrieval → Generation
Visual data → Visual/Cross-modal KG ┘
```

The difference lies in **how carefully** the fusion step is done, and **where each paper invests its effort**:

- **RAG-Anything**: Fusion via entity name matching is "good enough"; the real investment is in **cross-modal hybrid retrieval after fusion**
- **MMGraphRAG**: Fusion itself is the main event (SpecLink spectral clustering); the premise is that "cross-modal entity alignment is itself an unsolved problem"

---

## Paper 5｜VisRAG 2.0 / EVisRAG（2025）

### Current Problem

Existing Visual RAG (VRAG) methods have three core deficiencies in **multi-image** scenarios:

1. **Weak cross-image perception**: When multiple images are retrieved, the model struggles to reliably integrate fine-grained visual evidence across them
2. **Dependency on external agents**: Some methods introduce auxiliary agents to guide visual perception, increasing architectural complexity and instability
3. **Blurred credit assignment**: During training, reward signals for "perceptual ability" and "reasoning ability" are mixed together, interfering with each other and preventing either from being learned well

### Research Question

> **How can a single VLM, without relying on external tools, first collect evidence image-by-image and then reason to an answer, with distinct reward signals that train perception and reasoning independently?**

### Scope

Multi-image VRAG scenarios (visual QA over multiple retrieved pages); backbone model is Qwen2.5-VL-7B.

### Method & Highlights

**Two innovations: EVisRAG framework + RS-GRPO training method**

#### ① EVisRAG: Detective-Style Four-Stage Output Structure

The model is trained to generate responses in a fixed order:

```
[Observation] → Scan all retrieved images, write a coarse overall observation
      ↓
[Evidence]    → Read each image individually, record per-image relevant evidence
      ↓
[Reasoning]   → Cross-image reasoning over all collected evidence (hypothesize, cross-check)
      ↓
[Final Answer]
```

$$D_R \to r_{observe} \to r_{evidence} \to r_{reason} \to r_{answer}$$

**Key design details**:
- The evidence stage processes images **one at a time** (per-image); each image is re-examined against the original $d_i$ rather than relying on a text summary from the previous stage — preventing "hallucinated recall" from substituting for actually looking at the image
- The reasoning stage's conditioning input **still includes the original images $D_R$**, so the model can always refer back to the raw visuals rather than reasoning purely from prior text
- The entire pipeline is **built into a single model** (Qwen2.5-VL-7B) — no external agent or tool calls required

#### ② RS-GRPO: Rewards Precisely Scoped to Corresponding Token Ranges

Previous GRPO/PPO training typically assigns a single score to the whole output. The problem: the model cannot tell whether it failed because it "looked at the image poorly" or "reasoned incorrectly" — the two error signals are mixed and dilute each other.

RS-GRPO's core idea is to **partition the output sequence into 4 scopes**, each receiving only its corresponding reward signal:

| Reward | Token Scope | What It Checks |
|---|---|---|
| $R_{perception}$ | Observation + Evidence tokens | Whether per-image evidence captures the key content (compared against ground truth evidence generated by a large VLM) |
| $R_{derivation}$ | Reasoning + Answer tokens | Whether the final answer is correctly derived from the previously recorded evidence |
| $R_{format}$ | All scopes | Whether output follows the required "observe → evidence → reason → answer" structure |

$$\mathcal{M}(t) = \begin{cases} \{R_{perception}, R_{format}\} & t \in \mathcal{T}_{observe} \cup \mathcal{T}_{evidence} \\ \{R_{derivation}, R_{format}\} & t \in \mathcal{T}_{reason} \cup \mathcal{T}_{answer} \end{cases}$$

At each token position, multiple applicable rewards are averaged; this per-token reward is then normalized against other samples in the same group (standard GRPO advantage estimation), and the result feeds into a PPO-style clipped objective for optimization.

**Highlight**: This is like grading a math exam by scoring "working" and "final answer" separately — poor image perception deducts perception points, flawed reasoning deducts reasoning points, and the two signals never contaminate each other, so the model knows precisely what to improve.

**Training pipeline**: SFT warm-start first (so the model learns the output format), then RS-GRPO reinforcement learning to further improve both perceptual and reasoning quality.
