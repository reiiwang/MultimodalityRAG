# Multimodal RAG 論文筆記

> 整理日期：2026-06-25

---

## 目錄

1. [ColPali — Efficient Document Retrieval with Vision Language Models](#1-colpali)
2. [VLM2Vec-V2 — Advancing Multimodal Embedding for Videos, Images, and Visual Documents](#2-vlm2vec-v2)
3. [MMGraphRAG — Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs](#3-mmgraphrag)
4. [RAG-Anything — All-in-One RAG Framework](#4-rag-anything)
5. [VisRAG 2.0 — Evidence-Guided Multi-Image Reasoning in Visual RAG](#5-visrag-20)

---

## 1. ColPali

**論文**：ColPali: Efficient Document Retrieval with Vision Language Models
**arXiv**：2407.01449 | **發表**：ICLR 2025
**作者**：Manuel Faysse 等

### 1.1 動機與問題

傳統 document retrieval pipeline 依賴 OCR → text extraction → text embedding 的多步驟流程。這條路徑有三大缺點：
- 需要維護多個獨立元件（OCR engine、layout parser、chunker）
- 無法有效利用文件中的 **視覺線索**（圖表、表格排版、顏色標示）
- 對 PDF 中富含視覺資訊的頁面（如財報、投影片）效果差

### 1.2 核心方法

#### Architecture：三步驟編碼

ColPali 以 **PaliGemma-3B** 為基底，編碼流程分三步：

**Step 1 — SigLIP-So400m（patch 編碼）**
Vision Transformer 將頁面截圖切成 N 個 patch，每個 patch 各自編碼成一個高維向量。此時每個向量只「看到」自己那塊，缺乏跨 patch 的語境。

**Step 2 — Gemma 2B（語境化）**
N 個 patch 向量以 soft tokens 的形式送入 Gemma 2B。透過 attention 機制，每個 patch 向量能看到其他所有 patch——**Gemma 2B 在這裡不是在生成文字**，而是讓每個 patch 向量吸收整頁語境後，輸出帶有全局理解的新向量。

**Step 3 — Linear projection（壓縮至 128 維）**
Gemma 2B 輸出的向量維度極高（數千維），直接用於計算相似度的儲存成本大、速度慢。加一個可學習的矩陣 W，將每個向量壓縮到固定的 **128 維**，矩陣 W 在訓練時同步優化，確保壓縮後不損失重要語意。

最終，一頁文件 = **N 個 128 維向量**（multi-vector representation，不是文字、也不是單一全域向量）：
```
文件頁面截圖
    ↓ SigLIP 切 patch
[patch₁, ..., patchₙ]        ← N 個高維向量，無跨 patch 語境
    ↓ Gemma 2B（patch 互相 attend）
[帶語境的 patch₁, ..., patchₙ]
    ↓ Linear projection（矩陣 W）
[128維向量₁, ..., 128維向量ₙ]  ← 最終儲存的 multi-vector
```

#### Late Interaction（類 ColBERT 機制）與查詢流程

查詢時，query 文字同樣編碼成一串 **token 向量**（每個 token 一個向量）。

**每一頁**的分數計算方式（MaxSim）：
```
對每個 query token：
    掃描該頁所有 N 個 patch 向量
    取最高相似度（最相關的那個 patch）

總分 = 所有 query token 的最高分加總
```

```
Score(q, d) = Σ_{i ∈ query tokens} max_{j ∈ patch vectors} (q_i · d_j)
```

對**每一頁**都算出一個總分後，按分數排序，回傳分數最高的幾頁。**回傳單位是整頁**，不是某個 patch 也不是某段文字。

**多向量設計的優勢**：query 中「營收」這個 token 可精準對到圖表區塊的 patch，「成長率」可對到數字區塊的 patch，兩者的分數都加進來，這一頁的總分就很高。傳統 single-vector 把整頁壓成一個向量，細節容易被平均掉而淹沒在噪音中。

### 1.3 Benchmark：ViDoRe

作者提出 **ViDoRe（Visual Document Retrieval Benchmark）**，包含：
- 127,346 張高品質 PDF 頁面
- 涵蓋多個領域（科學、醫療、財務等）
- 評估指標：**nDCG@5**

**實驗結果**：
| 方法 | nDCG@5 |
|------|--------|
| BM25 | ~65 |
| BGE-M3（text-based） | ~75 |
| **ColPali** | **81.3** |

### 1.4 重要貢獻

1. 將 document retrieval 簡化為 **end-to-end trainable** 的視覺模型
2. 提出 ViDoRe benchmark，供後續研究比較
3. 證明視覺資訊對 document retrieval 的重要性

---

## 2. VLM2Vec-V2

**論文**：VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents
**arXiv**：2507.04590 | **發表**：2025 年 7 月
**作者**：Rui Meng 等（Salesforce Research、UC Santa Barbara、Waterloo、Tsinghua）

### 2.1 動機與問題

現有 multimodal embedding 模型（VLM2Vec v1、E5-V、GME）幾乎只聚焦在 **natural images**（MSCOCO、Flickr、ImageNet），缺乏對以下模態的支援：
- **Video**（時序內容、temporal grounding）
- **Visual documents**（PDF、投影片、表格、網頁）

這限制了在 AI agent、multimodal search、RAG 等應用場景的實用性，例如 YouTube 影片搜尋、PDF 文件查找。

---

### 2.2 核心方法

#### Section 3.1 — Backbone 選擇：Qwen2-VL

選用 **Qwen2-VL** 作為 backbone，理由有三：

| Qwen2-VL 特性 | 用途 |
|--------------|------|
| **Naive Dynamic Resolution** | 可彈性處理不同解析度的輸入（文件頁面往往比一般圖片需要更高解析度） |
| **M-RoPE（Multimodal Rotary Position Embedding）** | 捕捉視覺輸入的 spatial（2D 位置）與 temporal（時間序列）結構 |
| **統一 2D/3D convolution 架構** | 同一套架構可一致處理圖片（2D）和影片（3D），無需分開設計 |

#### Section 3.2 — Contrastive Learning（對比學習）

**Embedding 的取法**：將 query 和 target 分別送入同一個 VLM，取 **最後一層最後一個 token 的 hidden state** 作為 embedding 向量（`h_q`, `h_t`）。

**損失函數：InfoNCE Loss**（含 in-batch negatives + hard negatives）

$$\mathcal{L} = -\log \frac{\phi(h_q^{inst}, h_{t^+})}{\phi(h_q^{inst}, h_{t^+}) + \sum_{t^- \in \mathcal{N}} \phi(h_q^{inst}, h_{t^-})}$$

其中：
- $\phi(h_q, h_t) = \exp\!\left(\frac{1}{\tau} \cos(h_q, h_t)\right)$，溫度 $\tau = 0.02$
- $\mathcal{N}$ = 所有 negatives 的集合（in-batch 其他樣本 + 預先挖掘的 hard negatives）
- 越低溫度（小 $\tau$）→ 相似度分佈越尖銳 → 訓練信號越強

#### Section 3.3 — Instruction-Conditioned Embedding 格式

每個 training pair 都包含 **(query, positive target)**，並在 query 前加上 task-specific instruction：

```
[VISUAL TOKEN]  Instruct: {task instruction}
                Query: {q}
```

- `[VISUAL TOKEN]`：模態標記。圖片用 `<|image_pad|>`，影片用 `<|video_pad|>`
- `{task instruction}`：一句話描述任務，例如：
  - `"Find a video that contains the following visual content:"`（video retrieval）
  - `"Recognize the category of the video contents."`（video classification）
  - `"Retrieve the document page that answers the following question:"`（visual doc retrieval）
- Target 側也可加簡短 instruction，例如 `"Understand the content of the provided video:"`

這讓模型能依不同任務調整 embedding 的語義方向，實現跨任務泛化。

#### Section 3.4 — Data Sampling 策略

訓練資料來自異質來源，需要平衡採樣，論文設計了兩層策略：

**① On-the-fly Batch Mixing（動態批次混合）**
- 定義一張 **sampling weight table**，指定每個 dataset 的抽樣機率
- 每個 step 動態從不同資料集混抽，避免 overfitting 到單一模態

**② Interleaved Sub-batching（交錯子批次）**
- 將 batch（大小 1024）分割成若干 sub-batch（大小 64），每個 sub-batch 從同一來源抽樣
- 一個完整 batch = 16 個不同來源的 sub-batch 交錯
- **好處**：
  - sub-batch 內同質性高 → contrastive negatives 更 hard → 訓練更困難、更有效
  - 多個 sub-batch 交錯 → 全 batch 仍保有跨任務多樣性 → 優化穩定

消融實驗顯示 sub-batch size = 64 對 Image 任務最佳（倒 U 型曲線），而 Video/VisDoc 隨 sub-batch 增大持續提升。

---

### 2.3 Training Data 組成

| 資料來源 | 內容 | 數量 |
|---------|------|------|
| **LLaVA-Hound**（Zhang et al., 2024） | video-caption pairs + video QA（ChatGPT 合成） | 300k caption pairs + 240k QA pairs |
| **ViDoRe**（ColPali train set） | 視覺文件檢索訓練資料 | 118k |
| **VisRAG synthetic** | 合成文件問答 | 239k |
| **VisRAG in-domain** | 真實文件問答 | 123k |
| **MMEB-train** | 涵蓋 image classification、QA、retrieval、visual grounding 的圖文任務 | — |

Video 資料以 **8 幀均勻採樣（uniform frame sampling）** 表示一段影片，訓練和評估皆相同。

---

### 2.4 MMEB-V2 Benchmark 細節

在原版 MMEB（36 個 image/text 任務）基礎上新增 42 個任務，共 **9 個 meta-tasks、78 個 datasets**。

| Meta-task | 代表 datasets | 數量 |
|-----------|-------------|------|
| Image Classification | ImageNet-1K, ObjectNet, SUN397... | 10 |
| Image Retrieval | MSCOCO I2T/T2I, CIRR, FashionIQ... | 12 |
| Visual QA | DocVQA, ChartQA, TextVQA... | 10 |
| Visual Grounding | RefCOCO, Visual7W-Pointing... | 4 |
| **Video Classification** | Kinetics-700, HMDB51, UCF101... | 5 |
| **Video Retrieval** | MSR-VTT, DiDeMo, VATEX, YouCook2... | 5 |
| **Moment Retrieval** | QVHighlights, Charades-STA, MomentSeeker | 3 |
| **Video QA** | Video-MME, MVBench, NExT-QA, EgoSchema... | 5 |
| **Visual Document Retrieval** | ViDoRe×10, ViDoRe-V2×4, VisRAG×6, ViDoSeek×2, MMLongBench-Doc×2 | 24 |

評估指標：
- **Hit@1**：所有 image / video 任務（top-1 命中率）
- **NDCG@5**：所有 visual document retrieval 任務（與 ColPali 一致）

---

### 2.5 實驗結果

| 模型 | Image | Video | VisDoc | **All (78)** |
|------|-------|-------|--------|-------------|
| ColPali v1.3 (3B) | 34.9 | 28.2 | 71.0 | 44.4 |
| GME (2B) | 51.9 | 33.9 | 72.7 | 54.1 |
| GME (7B) | 56.0 | 38.6 | 75.2 | 57.8 |
| VLM2Vec-Qwen2VL (7B) | 65.5 | 34.0 | 46.4 | 52.3 |
| **VLM2Vec-V2 (2B)** | **64.9** | **34.9** | **65.4** | **58.0** |

關鍵發現：
- VLM2Vec-V2 (2B) 整體超越 GME (7B) 和 VLM2Vec (7B)，以更小參數量達到更好結果
- Image 任務接近 7B baseline；VisDoc 上仍不如專門優化的 ColPali
- 三種模態混合訓練（Image+VisDoc+Video）比任何兩種組合都好

### 2.6 Ablation 重點

**LoRA rank**：rank=16 最佳，增至 32 無額外收益（中等參數量對多模態泛化最有利）

**Training steps**：到 5K steps 仍未見飽和，VisDoc 和 Video 還有上升空間

**模態資料組合**：
| Training data | Image | VisDoc | Video | Avg |
|--------------|-------|--------|-------|-----|
| Image only | 62.5 | 41.5 | 31.5 | 45.2 |
| Image+Video | 63.3 | 51.9 | 29.7 | 48.3 |
| Image+VisDoc | 62.4 | 47.4 | 33.3 | 47.7 |
| **All three** | **62.7** | **52.2** | **32.4** | **49.1** |

### 2.7 重要貢獻

1. 第一個統一支援 text、image、video、visual document 的 embedding 模型
2. 提出 MMEB-V2（78 個 datasets）作為更全面的 multimodal embedding benchmark
3. Interleaved sub-batching 策略兼顧 contrastive hardness 與跨任務多樣性
4. LoRA 微調（2B 參數）以低成本超越多個 7B baseline

---

## 3. MMGraphRAG

**論文**：MMGraphRAG: Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs
**arXiv**：2507.20804 | **發表**：2025 年 7 月
**作者**：Xueyao Wan、Hang Yu 等

### 3.1 動機與問題

GraphRAG 透過 knowledge graph 結構化知識來改善 RAG，但現有方法仍 **text-centric**，原因在於：
- Multimodal Knowledge Graph（MMKG）難以自動建構
- 現有跨模態融合方法（shared embedding、captioning）需要 task-specific training
- 視覺結構知識（圖片中的空間關係）無法被保留
- 缺乏可解釋的 cross-modal reasoning path

### 3.2 核心方法

MMGraphRAG 的 pipeline 分為四個階段：

#### 階段一：Scene Graph 建構（視覺端）

使用 **YOLO** 偵測圖片中的 objects，再用 **MLLM（Multimodal LLM）** 推理物件間的關係，生成 **scene graph**（節點 = 物件，邊 = 關係）。

#### 階段二：Text Knowledge Graph 建構（文字端）

對文件文字部分進行 entity extraction 和 relation extraction，建立 text-based KG。

#### 階段三：SpecLink — 跨模態實體對齊

這是本文的核心創新：**SpecLink（Spectral Clustering-based Entity Linking）**

1. 計算 scene graph 中的 visual entities 與 text KG 中的 textual entities 之間的 embedding similarity
2. 以 **spectral clustering** 找出候選配對（能處理一對多、聚合相似概念）
3. 將兩個 KG 的節點融合，建立統一的 **MMKG（Multimodal Knowledge Graph）**

#### 階段四：Path-based Retrieval 與生成

根據 query 在 MMKG 上做 **path-based retrieval**（沿圖上的 reasoning paths 找出相關 subgraph），將 subgraph 作為 context 傳給 LLM 生成答案。推理路徑本身即為 **可解釋的推論依據**。

### 3.3 實驗結果

在 **DocBench** 和 **MMLongBench** 上達到 state-of-the-art，展現：
- 強大的跨 domain 適應性
- 清晰的 inference path（可解釋性高）

### 3.4 重要貢獻

1. 提出自動化的 MMKG 建構 pipeline
2. SpecLink 透過 spectral clustering 解決 cross-modal entity linking 難題
3. 透過 reasoning path 提供可解釋的多模態推理

---

## 4. RAG-Anything

**論文**：RAG-Anything: All-in-One RAG Framework
**arXiv**：2510.12323 | **發表**：2025 年 10 月
**作者**：Zirui Guo 等（HKUDS）
**開源**：https://github.com/HKUDS/RAG-Anything

### 4.1 動機與問題

真實世界的知識庫天生是 **multimodal** 的，包含：
- 文字段落
- 圖片、圖表
- 結構化表格
- 數學公式

然而現有 RAG framework 幾乎只能處理純文字，對 multimodal content 的支援極為有限，導致在富含圖表的長文件上效果很差。

### 4.2 核心方法

RAG-Anything 重新定義了 multimodal content 的表示方式：將不同模態的內容視為 **相互連結的知識實體（knowledge entities）**，而非孤立的資料類型。

#### 三大核心元件

**① Universal Indexing（通用索引）**
- 解析文件中的各類模態元素（文字、圖片、表格、公式）
- 統一表示為 knowledge entities，保留模態間的上下文關係

**② Dual Graph Construction（雙圖建構）**

這是 RAG-Anything 的核心架構：

| 圖結構 | 用途 |
|--------|------|
| **Cross-modal Knowledge Graph** | 捕捉不同模態間的語義關係（例如「圖一說明了文中的 X 概念」） |
| **Fine-grained Text Graph** | 保留細粒度的文字語義（段落、句子、詞彙層級的關聯） |

兩圖互補，共同表示文件知識。

**③ Cross-modal Hybrid Retrieval（跨模態混合檢索）**
- **Structural navigation**：沿 knowledge graph 的邊走訪，找出結構上相關的節點
- **Semantic matching**：傳統向量相似度搜尋
- 兩者融合，支援跨模態的複雜推理查詢

#### 生成端

將跨模態 retrieved context 組裝後傳入 LLM，生成融合多模態知識的答案。

### 4.3 實驗結果

- 在多個 multimodal benchmark 上超越 state-of-the-art
- 在 **長文件（long documents）** 上的提升尤為顯著（傳統方法在此場景嚴重退化）

### 4.4 重要貢獻

1. 首個統一處理文字、圖片、表格、公式的 all-in-one RAG framework
2. Dual graph 設計同時保留結構知識與文字語義
3. Cross-modal hybrid retrieval 兼顧結構導航與語義匹配

---

## 5. VisRAG 2.0

**論文**：VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation
**arXiv**：2510.09733 | **發表**：2025 年 10 月
**作者**：Yubo Sun、Chunyi Peng、Yukun Yan、Shi Yu、Zhenghao Liu、Chi Chen、Zhiyuan Liu、Maosong Sun
**開源**：https://github.com/OpenBMB/VisRAG

### 5.1 動機與問題

Visual RAG（VRAG）讓 Vision-Language Model（VLM）能利用外部視覺知識回答問題，但現有系統有一個根本缺陷：

**在多張圖片中無法可靠地感知與整合跨圖片的 evidence**

具體問題：
- VLM 在面對多圖輸入時，容易忽略部分圖片中的關鍵資訊
- 沒有明確的 evidence extraction 步驟，推理過程不透明
- 最終答案品質不穩定（hallucination 風險高）

### 5.2 核心方法

#### EVisRAG Framework（Evidence-Guided VisRAG）

將 VRAG 的推理過程拆解為兩個明確階段：

**階段一：Evidence Collection（per-image 證據收集）**
- 對每一張 retrieved image，VLM 逐一觀察並以 **自然語言記錄**「這張圖片對回答問題有什麼相關資訊」
- 輸出為 per-image evidence（每圖一份文字摘要）

**階段二：Evidence-Based Reasoning（基於證據的推理）**
- 將所有 per-image evidence 聚合
- 在 evidence 基礎上進行最終推理，得出答案

這個設計讓模型在回答問題之前，先 **強制感知所有相關圖片**，大幅減少遺漏關鍵視覺資訊的情況。

#### RS-GRPO 訓練方法

為訓練 EVisRAG，提出 **Reward-Scoped Group Relative Policy Optimization（RS-GRPO）**，是對 GRPO（Group Relative Policy Optimization）的改進版本：

- **Reward Scoping**：將細粒度的 reward 信號綁定到對應的 scope-specific tokens（evidence tokens vs. answer tokens），分別優化
- 對 **visual perception**（圖片理解準確度）和 **reasoning**（邏輯推理能力）進行 **聯合優化**
- 讓模型學會精確 localize 問題相關的 evidence，再進行 grounded reasoning

#### 對比傳統 GRPO

| | GRPO | RS-GRPO |
|---|------|---------|
| Reward 粒度 | 整體答案 | Scope-specific（分 evidence/answer） |
| 優化目標 | 答案正確性 | 感知準確度 + 推理正確性 |

### 5.3 實驗結果

以 **Qwen2.5-VL-7B** 為 backbone，EVisRAG 在多個 visual question answering benchmark 上：
- **Accuracy 平均提升 +19%**
- **F1 Score 平均提升 +27%**（相較於 backbone VLM）

### 5.4 重要貢獻

1. 提出 evidence-guided 的兩階段 VRAG 框架，使推理過程更透明
2. RS-GRPO 精確對齊 per-image evidence 的 reward，強化視覺感知能力
3. 在多個 benchmark 上以顯著幅度超越現有方法

---

## 綜合比較

| 論文 | 核心問題 | 方法亮點 | 適用場景 |
|------|---------|---------|---------|
| **ColPali** | Document retrieval 依賴 OCR | Multi-vector late interaction + VLM | PDF/投影片的視覺文件檢索 |
| **VLM2Vec-V2** | Embedding 只支援圖片 | 統一 Video/Image/Doc embedding | 多模態搜尋、RAG 索引 |
| **MMGraphRAG** | GraphRAG 是 text-only | MMKG + SpecLink 跨模態融合 | 含圖文的文件問答 |
| **RAG-Anything** | RAG 只能處理文字 | Dual Graph + Cross-modal retrieval | 複雜多模態文件的 RAG |
| **VisRAG 2.0** | Multi-image reasoning 不可靠 | Evidence-guided + RS-GRPO 訓練 | 需要整合多圖 evidence 的 VQA |

### 技術趨勢觀察

1. **從 text-centric 到真正 multimodal**：五篇論文都在解決 RAG/Retrieval pipeline 「只能處理文字」的限制
2. **Late interaction 的興起**：ColPali 和 VLM2Vec-V2 都採用 multi-vector representation，取代單一 global embedding
3. **Knowledge Graph 作為中介**：MMGraphRAG 和 RAG-Anything 都以 KG 作為跨模態知識的統一表示
4. **RL 強化推理**：VisRAG 2.0 引入 RS-GRPO，用強化學習訓練 VLM 的多圖感知與推理能力

---

## 參考來源

- [ColPali arXiv 2407.01449](https://arxiv.org/abs/2407.01449)
- [ColPali ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/99e9e141aafc314f76b0ca3dd66898b3-Paper-Conference.pdf)
- [VLM2Vec-V2 arXiv 2507.04590](https://arxiv.org/abs/2507.04590)
- [MMGraphRAG arXiv 2507.20804](https://arxiv.org/abs/2507.20804)
- [RAG-Anything arXiv 2510.12323](https://arxiv.org/abs/2510.12323)
- [VisRAG 2.0 arXiv 2510.09733](https://arxiv.org/abs/2510.09733)
