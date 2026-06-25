# 從純文字到真正的多模態 RAG

---

## 現實世界 RAG 的痛點

現今主流的檢索增強生成（RAG）系統大多侷限於「純文字」，但真實世界充滿了包含圖表、影片、數據與複雜排版的多模態資料。若依賴傳統的 OCR 或單純的圖片描述，會導致嚴重的語意與視覺線索遺失。

---

## 第一步 — 建立統一的多模態向量空間

### VLM2Vec-V2

**論文參考**：VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents

**核心痛點**：缺乏能同時處理文字、自然圖片、長影片與複雜視覺文件（如 PDF、簡報）的統一向量空間。現有的嵌入模型幾乎都只針對「自然圖像」與文字進行訓練，在面對影片或視覺文件時能力嚴重受限。

**突破性解法**：

- **MMEB-V2 基準測試**：作者擴展了評估框架，新增影片檢索、時間片段定位、影片分類、影片問答及視覺文件檢索等五類任務，共涵蓋 9 個元任務、78 個資料集。

- **通用嵌入模型 VLM2Vec-V2**：以 Qwen2-VL 為骨幹，選用原因有三：
  - **動態解析度（Naive Dynamic Resolution）**：能彈性處理任意解析度的輸入，文件頁面往往比一般圖片需要更高解析度才能讀清楚細節
  - **多模態旋轉位置編碼（M-RoPE）**：同時編碼視覺的空間位置（2D）與影片的時序位置，讓同一套位置編碼適用於圖片與影片
  - **統一 2D/3D 卷積架構**：圖片與影片共用同一套底層架構，不需要為不同模態設計不同的編碼器

  **向量取法**：query 和 target 分別送入模型後，取**最後一層最後一個 Token 的 hidden state** 作為該輸入的 embedding 向量。

  **對比學習損失函數（InfoNCE Loss）**：最大化「正確 query-target 對」的相似度，同時壓低「錯誤配對」的相似度。分母包含同批次的其他樣本（in-batch negatives）以及預先挖掘的 hard negatives，溫度參數 τ = 0.02（極低溫使分佈尖銳，讓訓練信號更強）。

  **任務指令格式（Instruction-Conditioned Embedding）**：每個 query 前加上一句話的任務描述，引導模型依任務調整向量語義的方向，例如影片檢索用「Find a video that contains the following visual content:」，視覺文件檢索用「Retrieve the document page that answers the following question:」：
  ```
  [視覺模態標記]  指令：{任務描述}
                 查詢：{問題}
  ```
  Target 側也可選擇性加上簡短指令，例如「Understand the content of the provided video:」，幫助模型從正確角度表示目標內容。

- **交錯子批次訓練（Interleaved Sub-batching）**：為了穩定訓練包含文字、圖片與影片的異質資料，將大批次（1024 筆）切分為子批次（各 64 筆），每個子批次來自同一資料來源。子批次內部同質性高，使對比學習難度上升；跨子批次交錯則保有全批次的多樣性，有效提升跨模態的泛化能力。

**成效數據**：VLM2Vec-V2（20 億參數）在 78 個資料集上的整體平均分達到 **58.0**，超越 GME-7B（57.8）和 VLM2Vec-7B（52.3），以更小的模型規模達到更好的結果。

---

## 第二步 — 跳過 OCR 的視覺文件索引

### ColPali

**論文參考**：ColPali: Efficient Document Retrieval with Vision Language Models

**核心痛點**：傳統系統在建構索引時，需要經過版面偵測、OCR 擷取與切塊等繁瑣步驟，不僅容易遺失圖表排版資訊，處理一頁甚至高達 **7.22 秒**。

**突破性解法**：

- **將文件視為圖像**：ColPali 直接對「文件頁面的截圖」進行嵌入與檢索，完全跳過 OCR。視覺語言模型 PaliGemma-3B 中的視覺編碼器（SigLIP-So400m）將頁面切成圖像區塊，語言模型（Gemma 2B）對區塊嵌入進行語境化，再透過線性投影壓縮至 128 維向量空間。 
  整個編碼過程分三步：

  **Step 1 — SigLIP 切 patch、編碼**：Vision Transformer（SigLIP-So400m）將頁面截圖切成 N 個小區塊（patch），每個 patch 各自編碼成一個高維向量。此時每個向量只「看到」自己那塊，缺乏整頁的語境。

  **Step 2 — Gemma 2B 語境化**：這 N 個 patch 向量以「soft tokens」的形式送入語言模型 Gemma 2B。透過 attention 機制，每個 patch 向量能看到其他所有 patch，就像語言模型讀句子時每個字會參考前後文一樣。**Gemma 2B 在這裡不是在生成文字**，而是讓每個 patch 向量吸收整頁語境後，輸出帶有全局理解的新向量。

  **Step 3 — 線性投影壓縮至 128 維**：Gemma 2B 輸出的向量維度極高（數千維），直接用於計算相似度儲存成本大、速度慢。因此加一個 linear projection layer（一個可學習的矩陣 W），將每個向量壓縮到固定的 128 維。這個矩陣在訓練時同步優化，確保壓縮後不損失重要語意。

  最終，一頁文件 = **N 個 128 維向量**（而非一段文字、也非單一全域向量）：
  ```
  文件頁面截圖
      ↓ SigLIP 切 patch
  [patch₁, patch₂, ..., patchₙ]  ← N 個高維向量
      ↓ Gemma 2B 語境化（patch 互相看到彼此）
  [帶語境的patch₁, ..., 帶語境的patchₙ]
      ↓ Linear projection（矩陣 W）
  [128維向量₁, 128維向量₂, ..., 128維向量ₙ]  ← 最終儲存的 multi-vector
  ```

- **後期交互機制（Late Interaction / MaxSim）**：查詢時，query 文字也被編碼成一串 token 向量（每個詞一個向量），然後對**每一頁**文件計算分數：

  ```
  對每個 query token：
      掃描該頁所有 N 個 patch 向量
      取最高相似度（最相關的那個 patch）

  該頁總分 = 所有 query token 的最高分加總
  ```

  對所有候選頁面都算出總分後，按分數排序，回傳分數最高的幾頁。**回傳的單位是整頁**，不是某個 patch，也不是某段文字。

  這個設計的核心優勢：query 中「營收」這個 token 可以精準對到頁面圖表區塊的 patch，「成長率」可以對到數字區塊的 patch，兩者分數都加進來，讓這一頁的總分很高。相比之下，傳統 single-vector 把整頁壓成一個向量，細節容易被平均掉而淹沒在噪音中。

  查詢時的完整流程
  Query 端
  你的問題（例如「這份報告的營收成長率是多少？」）也會過一個 text encoder，變成一串 token 向量（每個詞/字一個向量）。
  比對端
  每一頁文件都存著 N 個 128 維的 patch 向量。
  計分方式（MaxSim）：
  對每個 query token：
      去掃所有 N 個 patch 向量
      找出跟自己最像的那個 patch → 取那個最高相似度分數
  最後把所有 query token 的最高分加總 → 這一頁的總分
  最後排名
  對每一頁都算出一個總分，然後按分數高低排序，回傳分數最高的那幾頁。
  所以你理解的完全正確：回傳的單位是「整頁」，不是某個 patch，也不是某段文字。
  多向量設計的好處是：
  - query 裡問「營收」的 token 可以精準對到頁面中圖表那塊的 patch
  - query 裡問「成長率」的 token 可以對到另一塊有數字的 patch
  - 最後兩者的分數都加進來，讓這一頁的總分很高
  而傳統 single-vector（把整頁壓成一個向量）的問題是：整頁的資訊被平均掉了，細節很容易淹沒在噪音裡。


- **ViDoRe 基準測試**：新提出的視覺文件檢索基準，包含 127,346 張 PDF 頁面，涵蓋科學、醫療、財務等多個領域，以 nDCG@5 作為評估指標。

**成效數據**：

| 方法 | nDCG@5 |
|------|--------|
| BM25 | 約 65 |
| BGE-M3（純文字方法） | 約 75 |
| **ColPali** | **81.3** |

離線建立索引的時間從每頁 7.22 秒大幅縮短至 **0.39 秒**；在富含圖表與複雜排版的文件中，檢索準確率顯著超越所有純文字方法。

---

## 第三步 — 以知識圖譜進行跨模態推理

### RAG-Anything 與 MMGraphRAG

**論文參考**：
- RAG-Anything: All-in-One RAG Framework
- MMGraphRAG: Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs

**核心痛點**：有了精準的檢索能力後，面對需要跨越不同模態的複雜邏輯問題（例如：文字敘述結合圖表數據），將資料打散成扁平向量是不夠的，需要將其「結構化」。傳統知識圖譜系統無法處理圖片；而傳統跨模態對齊（如圖片轉文字描述）又會遺失細粒度的視覺結構知識。

---

#### RAG-Anything（宏觀架構）

框架由三個核心元件組成：**通用多模態索引、跨模態混合檢索、知識增強生成**。

---

**元件一：通用多模態知識表示（Universal Representation）**

文件解析時，RAG-Anything 用不同的專門解析器拆解各類模態內容，保留其結構語境：

| 模態 | 解析方式 |
|------|---------|
| 文字 | 切成段落 / 列表項目，保留層次結構 |
| 圖片 | 提取圖片 + caption + 跨引用資訊 |
| 表格 | 解析成 header、row、cell 的結構化節點 |
| 數學公式 | 轉成 LaTeX 符號表示 |

每個解析出來的內容單元 `c_j = (模態類型, 原始內容)`，並確保圖片與 caption 保持連結、公式與周圍定義保持連結、表格與說明文字保持連結——而非把所有東西壓成扁平文字。

---

**元件二：雙圖建構（Dual-Graph Construction）**

單張圖要同時處理所有模態往往會丟失各模態特有的結構訊號，因此 RAG-Anything 建構兩張互補的圖：

**① Cross-Modal Knowledge Graph（跨模態知識圖譜）**

針對圖片、表格、公式等非文字內容：
1. 用 MLLM 為每個非文字單元生成兩種文字表示：「詳細描述（用於跨模態檢索）」和「實體摘要（包含名稱、類型、屬性，用於圖建構）」
2. 對詳細描述進行 entity extraction + relation extraction，取得該單元內部的實體與關係
3. 以「多模態實體節點」作為 anchor，內部細粒度實體透過 `belongs_to` 邊與 anchor 連結
4. 最終保留視覺內容的空間結構（如多面板圖的 panel→caption→axis 邊、表格的 row→cell→unit 邊）

**② Text-Based Knowledge Graph（文字知識圖譜）**

對文字段落做傳統的 entity recognition + relation extraction（類似 LightRAG / GraphRAG 的做法），取得詞彙層級的實體關係。

**③ 圖融合（Graph Fusion）**

以 entity name 為主鍵，對齊兩張圖中的同名實體後合併，形成統一的知識圖譜 G = (V, E)。再將圖中所有實體、關係、內容單元都編碼為密集向量，建立 embedding table T。最終 index = (G, T)，同時具備結構化知識與向量語義搜尋能力。

---

**元件三：跨模態混合檢索（Cross-Modal Hybrid Retrieval）**

查詢時分兩條路徑並行：

**① 結構導航（Structural Knowledge Navigation）**
- 對 query 做關鍵詞匹配與 entity recognition，找到圖上相關節點
- 從相關節點出發做 neighborhood expansion（多跳走訪），收集相連的實體、關係、內容單元
- 特別擅長捕捉「跨模態的多跳推理」（例如：文字提到的圖表 → 圖表內部的 panel → panel 上的 y 軸標籤）

**② 語義相似度匹配（Semantic Similarity Matching）**
- 將 query 編碼成 embedding，對 embedding table T 做 cosine similarity 搜尋
- 回傳語義最相關的 top-k 內容單元（包含文字、圖片、表格、公式）
- 能找到結構上不相連但語義相關的內容

**③ 多信號融合評分（Multi-Signal Fusion Scoring）**

兩條路徑的候選集合合併後，依三種信號加權評分排序：
- 圖結構重要性（graph topology 中的中心度）
- 語義相似度分數
- Query 推斷的模態偏好（例如 query 含「圖表」→ 提升圖片類內容的權重）

---

**元件四：生成（Synthesis）**

從 retrieved candidates 中：
1. 組裝結構化文字 context（實體摘要 + 關係描述 + chunk 內容）
2. 對圖片、表格等視覺類 chunk 做 dereferencing，還原原始視覺內容
3. 同時將文字 context 與視覺內容輸入 VLM 生成答案

---

**成效數據**

在 DocBench（229 份文件，平均 66 頁）和 MMLongBench（135 份文件，平均 47.5 頁）上的準確率：

| 方法 | DocBench | MMLongBench |
|------|---------|------------|
| GPT-4o-mini | 51.2% | 33.5% |
| LightRAG | 58.4% | 38.9% |
| MMGraphRAG | 61.0% | 37.7% |
| **RAG-Anything** | **63.4%** | **42.8%** |

**長文件優勢更顯著**：文件超過 100 頁時，RAG-Anything 與 MMGraphRAG 的差距擴大到 13 個百分點（68.2% vs 54.6%）。

**兩個具體案例說明結構化圖譜的優勢**：
- **多面板圖理解**：query 問「哪個模型的 style space 分群更清晰？」，其他方法錯把相鄰的 content-space panel 當成 style-space panel，RAG-Anything 的圖結構明確區分了兩個 panel，答對了（DAE，而非 VAE）
- **財務表格導航**：query 問「2020 年 Novo Nordisk 的薪資支出是多少？」，其他方法把相鄰的「Share-based payments」列或不同年份的數字混淆了，RAG-Anything 的 row→cell→unit 圖結構精準導向正確格（DKK 26,778 million）

---

#### MMGraphRAG（微觀對齊）

**將視覺節點視為圖譜中的一等公民**：

1. **場景圖建構**：使用 YOLO 偵測圖片中的物件，再以多模態大語言模型推理物件間的空間關係，生成場景圖（節點＝物件，邊＝關係）。
2. **文字知識圖譜建構**：對文件文字進行實體抽取與關係抽取。
3. **SpecLink（光譜聚類實體連結）**：核心創新。傳統最近鄰搜尋在處理一對多的視覺文字對應時容易出錯。SpecLink 計算視覺實體與文字實體之間的嵌入相似度矩陣，再透過光譜聚類找出候選對齊群組，最終將兩張圖譜融合為統一的多模態知識圖譜。
4. **路徑檢索**：根據查詢在多模態知識圖譜上沿推理路徑走訪，找出相關子圖作為生成的上下文——推理路徑本身即為**可解釋的推論依據**。

**成效數據**：在 DocBench 和 MMLongBench 上達到最先進水準，並具備清晰可追溯的推理路徑。

---

## 第四步 — 引導視覺語言模型主動搜尋視覺證據

### VisRAG 2.0（EVisRAG）

**論文參考**：VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation

**核心痛點**：檢索系統找到正確圖片後餵給生成模型，任務還沒結束。模型面對多張圖片時，經常只「看見」部分圖片的內容，無法精準整合跨圖片的視覺證據，最終導致幻覺或答非所問。

**突破性解法**：

- **證據引導的兩階段推理（EVisRAG）**：強制模型遵循像偵探一樣的明確步驟：
  - **第一階段——逐圖證據收集**：模型逐一觀察每張檢索到的圖片，並用自然語言寫下「這張圖片對本問題的相關資訊」，確保不遺漏任何圖片中的關鍵視覺資訊。
  - **第二階段——基於證據推理**：將所有逐圖證據彙總，模型在明確的書面證據基礎上推導最終答案，推理過程完全透明可追溯。

- **RS-GRPO（作用域限定的群組相對策略優化）**：傳統強化學習只對整體答案給予獎勵，無法驅動精確的視覺證據定位。RS-GRPO 將獎勵精確綁定到對應的 Token 範圍——證據 Token 和答案 Token 分別受到各自的細粒度獎勵監督，聯合優化視覺感知準確度與推理正確性。

**成效數據**（以 Qwen2.5-VL-7B 為骨幹）：

| 指標 | 提升幅度 |
|------|---------|
| 準確率（Accuracy） | 平均 **+19%** |
| F1 分數 | 平均 **+27%** |

---

## 綜合比較

| 論文 | 解決的層次 | 核心技術 | 主要基準測試 |
|------|-----------|---------|------------|
| **VLM2Vec-V2** | 表示（Representation） | Qwen2-VL + 任務指令嵌入 + 交錯子批次訓練 | MMEB-V2（78 個任務） |
| **ColPali** | 索引（Indexing） | 多向量後期交互（MaxSim）+ 視覺區塊嵌入 | ViDoRe（nDCG@5） |
| **RAG-Anything** | 推理（宏觀） | 雙圖建構 + 跨模態混合檢索 | 多模態長文件基準 |
| **MMGraphRAG** | 推理（微觀） | 場景圖 + SpecLink（光譜聚類）+ 路徑檢索 | DocBench、MMLongBench |
| **VisRAG 2.0** | 生成（Generation） | 證據引導兩階段推理 + RS-GRPO | 多個視覺問答基準 |

**技術趨勢**：
1. **告別 OCR**：ColPali 和 VLM2Vec-V2 都證明直接處理視覺輸入比 OCR 轉文字更有效
2. **多向量取代單一向量**：後期交互機制能捕捉全域嵌入遺漏的細粒度視覺語意匹配
3. **知識圖譜作為跨模態橋樑**：RAG-Anything 和 MMGraphRAG 都以知識圖譜統一不同模態的知識
4. **強化學習驅動推理**：VisRAG 2.0 用 RS-GRPO 讓模型主動搜尋視覺證據，而非被動接受多圖輸入

---

## 參考論文

| 論文 | arXiv | 年份 |
|------|-------|------|
| ColPali: Efficient Document Retrieval with Vision Language Models | [2407.01449](https://arxiv.org/abs/2407.01449) | ICLR 2025 |
| VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents | [2507.04590](https://arxiv.org/abs/2507.04590) | 2025.07 |
| MMGraphRAG: Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs | [2507.20804](https://arxiv.org/abs/2507.20804) | 2025.07 |
| RAG-Anything: All-in-One RAG Framework | [2510.12323](https://arxiv.org/abs/2510.12323) | 2025.10 |
| VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual RAG | [2510.09733](https://arxiv.org/abs/2510.09733) | 2025.10 |
