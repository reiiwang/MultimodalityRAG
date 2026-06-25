# From Text-Only to Truly Multimodal RAG

---

## The Pain Points of Real-World RAG

Most mainstream Retrieval-Augmented Generation (RAG) systems are limited to plain text, yet the real world is filled with multimodal data containing charts, videos, structured tables, and complex layouts. Relying on traditional OCR or simple image captioning leads to severe loss of semantic and visual cues.

---

## Step 1 — Creating a Unified Multimodal Embedding Space

### VLM2Vec-V2

**Reference**: VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents

**Core Pain Point**: Existing embedding models are trained almost exclusively on natural images and text, making them incapable of handling videos or visual documents (PDFs, slides, web screenshots). A unified vector space that spans text, images, long videos, and complex visual documents simply does not exist.

**Breakthroughs**:

- **MMEB-V2 Benchmark**: The authors extended the evaluation framework with five new task types — video retrieval, moment retrieval, video classification, video question answering, and visual document retrieval — covering 78 datasets across 9 meta-tasks.

- **Unified Embedding Model (VLM2Vec-V2)**: Built on Qwen2-VL as the backbone, chosen for three key reasons:
  - **Naive Dynamic Resolution**: Handles inputs at any resolution — document pages require significantly higher resolution than natural images to preserve fine-grained text and layout details
  - **M-RoPE (Multimodal Rotary Position Embedding)**: Encodes both spatial position (2D, for images) and temporal position (for videos) within a single positional encoding scheme, making the same architecture work across modalities
  - **Unified 2D/3D convolution architecture**: Images and videos share the same underlying encoder — no need to design separate encoders per modality

  **Embedding extraction**: Both query and target are passed independently through the model; the **last token's hidden state in the final layer** is taken as the embedding vector for that input.

  **Contrastive loss (InfoNCE)**: Maximizes similarity between correct query-target pairs while pushing down similarity for incorrect pairings. The denominator includes in-batch negatives (other samples in the same batch) and pre-mined hard negatives. Temperature τ = 0.02 (very low, sharpening the similarity distribution for stronger training signal).

  **Instruction-Conditioned Embedding format**: Each query is prepended with a one-sentence task description that steers the embedding toward the right semantic direction for the task — e.g., video retrieval uses `"Find a video that contains the following visual content:"`, visual document retrieval uses `"Retrieve the document page that answers the following question:"`:
  ```
  [VISUAL TOKEN]  Instruct: {task instruction}
                  Query: {q}
  ```
  The target side can also optionally receive a short instruction (e.g., `"Understand the content of the provided video:"`) to guide how the target content is represented.

- **Interleaved Sub-batching**: To stabilize contrastive training over heterogeneous data sources (text, image, video), the full batch (size 1024) is divided into sub-batches (size 64), each drawn from a single source. Intra-sub-batch homogeneity raises the difficulty of contrastive discrimination, while interleaving multiple sub-batches preserves cross-task diversity across the full batch.

**Results**: VLM2Vec-V2 (2B parameters) achieves an overall average of **58.0** across 78 datasets, outperforming GME-7B (57.8) and VLM2Vec-7B (52.3) with a significantly smaller model.

---

## Step 2 — Bypassing OCR for Visual Document Indexing

### ColPali

**Reference**: ColPali: Efficient Document Retrieval with Vision Language Models

**Core Pain Point**: Traditional RAG pipelines for documents require slow and brittle multi-step processing — layout detection, OCR extraction, and chunking. This not only discards visual cues (charts, colors, layout) but also takes up to **7.22 seconds per page** to index.

**Breakthroughs**:

- **Treating Documents as Images**: ColPali directly embeds document page screenshots using **PaliGemma-3B**, completely bypassing OCR. The encoding process has three steps:

  **Step 1 — SigLIP patch encoding**: The Vision Transformer (SigLIP-So400m) splits the page screenshot into N patches and encodes each into a high-dimensional vector independently. At this point each vector only "sees" its own patch — no cross-patch context yet.

  **Step 2 — Gemma 2B contextualization**: The N patch vectors are fed into Gemma 2B as soft tokens. Through attention, each patch vector can attend to all other patches — just like a language model reads a sentence where each word attends to surrounding words. **Gemma 2B is not generating text here**; it outputs new vectors for each patch that now carry page-level context.

  **Step 3 — Linear projection to 128 dimensions**: Gemma 2B outputs very high-dimensional vectors (thousands of dims), which are expensive to store and slow to compare at scale. A learnable linear projection layer (matrix W) compresses each vector to a fixed 128 dimensions. W is optimized during training to preserve semantically important information.

  The final representation of one page = **N vectors of 128 dimensions** (not text, not a single global vector):
  ```
  Document page screenshot
      ↓ SigLIP patch encoding
  [patch₁, patch₂, ..., patchₙ]  ← N high-dim vectors
      ↓ Gemma 2B contextualization (patches attend to each other)
  [contextualized patch₁, ..., contextualized patchₙ]
      ↓ Linear projection (matrix W)
  [128-dim vector₁, 128-dim vector₂, ..., 128-dim vectorₙ]  ← stored multi-vector
  ```

- **Late Interaction (MaxSim)**: At query time, the query text is also encoded into a sequence of token vectors (one per word). For **every candidate page**, a score is computed:

  ```
  For each query token:
      Scan all N patch vectors of the page
      Take the highest similarity score (the most relevant patch)

  Page score = sum of the highest scores across all query tokens
  ```

  All candidate pages are scored this way, ranked, and the top pages are returned. **The retrieval unit is a whole page** — not a patch, not a text chunk.

  $$\text{Score}(q, d) = \sum_{i \in \text{query tokens}} \max_{j \in \text{patch vectors}} (q_i \cdot d_j)$$

  The key advantage: the query token "revenue" can precisely match the patch covering the chart, while "growth rate" matches the patch with the numbers — both scores contribute to the page total. A traditional single-vector approach averages the entire page into one vector, drowning out fine-grained details in noise.

- **ViDoRe Benchmark**: A new Visual Document Retrieval Benchmark with 127,346 high-quality PDF pages across multiple domains (science, medicine, finance), evaluated by nDCG@5.

**Results**:

| Method | nDCG@5 |
|--------|--------|
| BM25 | ~65 |
| BGE-M3 (text-based) | ~75 |
| **ColPali** | **81.3** |

Indexing time reduced from 7.22s to **0.39s per page**; retrieval accuracy significantly surpasses all text-based methods on visually rich documents.

---

## Step 3 — Cross-Modal Reasoning with Knowledge Graphs

### RAG-Anything & MMGraphRAG

**References**:
- RAG-Anything: All-in-One RAG Framework
- MMGraphRAG: Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs

**Core Pain Point**: Precise retrieval is not enough for complex questions that require relational reasoning across modalities — for instance, correlating a textual description with data in a chart. Flat vector search cannot capture these structural relationships. Traditional GraphRAG is text-only, while naive cross-modal alignment (e.g., image captioning) discards fine-grained visual structure.

---

#### RAG-Anything (Macro Architecture)

The framework has three core components: **universal multimodal indexing, cross-modal hybrid retrieval, and knowledge-enhanced generation**.

---

**Component 1 — Universal Multimodal Representation**

RAG-Anything uses specialized parsers to decompose each document into typed content units, preserving structural context:

| Modality | Parsing approach |
|----------|----------------|
| Text | Segmented into coherent paragraphs / list items with hierarchy |
| Images | Extracted with captions, metadata, and cross-references |
| Tables | Parsed into structured header → row → cell → unit nodes |
| Equations | Converted to LaTeX symbolic representations |

Each unit `c_j = (modality_type, content)` preserves its connections to surrounding context — figures stay linked to captions, equations stay linked to nearby definitions, tables stay linked to explanatory text — rather than flattening everything into plain text.

---

**Component 2 — Dual-Graph Construction**

A single unified graph risks losing modality-specific structural signals, so RAG-Anything builds two complementary graphs:

**① Cross-Modal Knowledge Graph**

For non-text content (images, tables, equations):
1. An MLLM generates two textual representations per unit: a *detailed description* (optimized for retrieval) and an *entity summary* (name, type, attributes for graph construction)
2. Entity + relation extraction is run on the detailed description to extract fine-grained intra-chunk entities
3. A multimodal anchor node is created for each unit; intra-chunk entities connect to it via `belongs_to` edges
4. Visual structure is preserved: multi-panel figures get panel→caption→axis edges; tables get row→cell→unit edges

**② Text-Based Knowledge Graph**

Standard named entity recognition + relation extraction on text chunks (following LightRAG / GraphRAG methodology), capturing lexical-level entity relationships.

**③ Graph Fusion**

The two graphs are merged using entity names as alignment keys. All entities, relations, and content chunks are then encoded into dense vectors to form embedding table T. The final index = (G, T), supporting both structural traversal and vector similarity search.

---

**Component 3 — Cross-Modal Hybrid Retrieval**

Two parallel retrieval pathways run simultaneously:

**① Structural Knowledge Navigation**
- Keyword matching and entity recognition locate relevant nodes in the unified graph G
- Neighborhood expansion (multi-hop traversal) collects related entities, relations, and content chunks
- Particularly effective for cross-modal multi-hop reasoning (e.g., text reference → figure → specific panel → axis label)

**② Semantic Similarity Matching**
- Query is encoded into an embedding and compared against all entries in T via cosine similarity
- Returns top-k semantically similar chunks across all modalities (text, images, tables, equations)
- Surfaces semantically relevant content even when it lacks direct graph connections to the query

**③ Multi-Signal Fusion Scoring**

Results from both pathways are merged and ranked by three signals:
- Graph topology importance (centrality within G)
- Semantic similarity score
- Query-inferred modality preference (e.g., a query mentioning "figure" → upweights visual content)

---

**Component 4 — Synthesis**

From the top-ranked candidates:
1. Build structured textual context (entity summaries + relation descriptions + chunk contents)
2. Dereference visual chunks to recover original images/tables
3. Feed both textual context and visual content into a VLM for grounded answer generation

---

**Results**

Accuracy on DocBench (229 docs, avg. 66 pages) and MMLongBench (135 docs, avg. 47.5 pages):

| Method | DocBench | MMLongBench |
|--------|---------|------------|
| GPT-4o-mini | 51.2% | 33.5% |
| LightRAG | 58.4% | 38.9% |
| MMGraphRAG | 61.0% | 37.7% |
| **RAG-Anything** | **63.4%** | **42.8%** |

**Advantage grows with document length**: for documents over 100 pages, the gap over MMGraphRAG expands to 13 points (68.2% vs. 54.6%).

**Two case studies illustrate the value of structure-aware graphs**:
- **Multi-panel figure**: asked "which model's style space shows clearer cluster separation?", other methods confused the adjacent content-space panel with the style-space panel. RAG-Anything's graph explicitly distinguished the two panels and answered correctly (DAE, not VAE).
- **Financial table navigation**: asked "what was Novo Nordisk's total wages and salaries in 2020?", other methods confused adjacent rows ("Share-based payments") or wrong year columns. RAG-Anything's row→cell→unit graph navigated precisely to the correct cell (DKK 26,778 million).

---

#### MMGraphRAG (Micro Alignment)

**Visual nodes as first-class citizens in the knowledge graph**:

1. **Scene Graph Construction**: YOLO detects objects in images; an MLLM infers spatial relationships between them, producing a scene graph (nodes = objects, edges = relations).
2. **Text Knowledge Graph**: Standard entity and relation extraction from document text.
3. **SpecLink (Spectral Clustering-based Entity Linking)**: The key innovation. Traditional nearest-neighbor matching fails for one-to-many visual-textual correspondences. SpecLink computes an embedding similarity matrix between visual and textual entities, applies spectral clustering to find candidate alignment groups, and merges the two graphs into a unified **MMKG (Multimodal Knowledge Graph)**.
4. **Path-based Retrieval**: Queries traverse reasoning paths in the MMKG to retrieve relevant subgraphs as context — the paths themselves serve as **interpretable reasoning evidence**.

**Results**: State-of-the-art on DocBench and MMLongBench, with clear and interpretable inference paths.

---

## Step 4 — Guiding VLMs to Search for Visual Evidence

### VisRAG 2.0 (EVisRAG)

**Reference**: VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation

**Core Pain Point**: Even after the retrieval system finds the correct images and feeds them to the generative model, VLMs frequently hallucinate or miss key information when processing multiple images simultaneously — they fail to reliably perceive and integrate evidence across all retrieved images.

**Breakthroughs**:

- **Evidence-Guided Two-Stage Reasoning (EVisRAG)**: Forces the model to follow an explicit detective-style workflow:
  - **Stage 1 — Per-Image Evidence Collection**: The model observes each retrieved image one by one and records in natural language what information it contains relevant to the question.
  - **Stage 2 — Evidence-Based Reasoning**: The model aggregates all per-image evidence and derives the final answer from the collected evidence. The reasoning process is fully transparent and traceable.

- **RS-GRPO (Reward-Scoped Group Relative Policy Optimization)**: Standard GRPO rewards only the final answer, providing no fine-grained signal for evidence localization. RS-GRPO binds rewards to scope-specific tokens — evidence tokens and answer tokens receive separate fine-grained rewards — jointly optimizing both visual perception accuracy and reasoning correctness.

**Results** (backbone: Qwen2.5-VL-7B):

| Metric | Improvement |
|--------|------------|
| Accuracy | **+19% average** |
| F1 Score | **+27% average** |

---

## Summary

| Paper | Layer | Core Technique | Benchmark |
|-------|-------|---------------|-----------|
| **VLM2Vec-V2** | Representation | Qwen2-VL + Instruction embedding + Interleaved sub-batching | MMEB-V2 (78 tasks) |
| **ColPali** | Indexing | Multi-vector late interaction (MaxSim) + VLM patch embedding | ViDoRe (nDCG@5) |
| **RAG-Anything** | Reasoning (macro) | Dual-graph + Cross-modal hybrid retrieval | Multimodal long-doc benchmarks |
| **MMGraphRAG** | Reasoning (micro) | Scene graph + SpecLink (spectral clustering) + Path retrieval | DocBench, MMLongBench |
| **VisRAG 2.0** | Generation | Evidence-guided two-stage reasoning + RS-GRPO | Multiple VQA benchmarks |

**Key Trends**:
1. **Goodbye OCR**: Both ColPali and VLM2Vec-V2 demonstrate that directly processing visual input outperforms OCR-to-text pipelines
2. **Multi-vector over single-vector**: Late interaction captures fine-grained visual-semantic matching that global embeddings miss
3. **Knowledge Graphs as cross-modal bridges**: Both RAG-Anything and MMGraphRAG use KGs to unify knowledge across modalities
4. **RL-driven reasoning**: VisRAG 2.0 uses RS-GRPO to teach VLMs to actively search for visual evidence rather than passively consuming multi-image input

---

## References

| Paper | arXiv | Year |
|-------|-------|------|
| ColPali: Efficient Document Retrieval with Vision Language Models | [2407.01449](https://arxiv.org/abs/2407.01449) | ICLR 2025 |
| VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents | [2507.04590](https://arxiv.org/abs/2507.04590) | 2025.07 |
| MMGraphRAG: Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs | [2507.20804](https://arxiv.org/abs/2507.20804) | 2025.07 |
| RAG-Anything: All-in-One RAG Framework | [2510.12323](https://arxiv.org/abs/2510.12323) | 2025.10 |
| VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual RAG | [2510.09733](https://arxiv.org/abs/2510.09733) | 2025.10 |
