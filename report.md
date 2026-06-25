## The Pain Points of Real-World RAG 
現今主流的檢索增強生成 (RAG) 系統大多侷限於「純文本」，但真實世界充滿了包含圖表、影片、數據與複雜排版的多模態資料
。若依賴傳統的 OCR 或單純的圖片描述，會導致嚴重的語意與視覺線索遺失

--- 

Creating a Unified Multimodal Embedding Space
VLM2Vec-V2
To process multimodal data, strong underlying representation capabilities are required first. Most existing multimodal models are only trained on "natural images" and text, making them unable to handle videos or visual documents.
要處理多模態資料，首要任務是賦予模型跨越不同資料格式的「統一理解能力」。現有的嵌入模型大多只針對「自然圖像」與文本進行訓練，在面對影片或視覺文件時能力受限。
論文參考：VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents
核心痛點：缺乏能同時處理文字、自然圖片、長影片與複雜視覺文件（如 PDF、簡報）的統一向量空間。
突破性解法：
MMEB-V2 基準測試：作者擴展了評估框架，新增影片檢索、時間片段定位、影片分類、影片問答及視覺文件檢索等任務。
通用嵌入模型 VLM2Vec-V2：以 Qwen2-VL 為骨幹，透過加入任務指令 (Instruction-guided) 來引導模型學習。
交錯子批次訓練 (Interleaved sub-batching)：為了穩定訓練包含文字、圖片與影片的異質資料，將不同來源的資料分割成子批次並交錯組合，有效提升了跨模態的泛化能力。


ColPali
With the foundational model in place, when facing visual documents like PDFs, traditional RAG pipelines still rely on slow and error-prone PDF parsing, OCR, and layout detection.
具備了底層向量能力後，我們面臨的下一個瓶頸是「傳統 PDF 解析流程過於緩慢且易碎」。
論文參考：ColPali: EFFICIENT DOCUMENT RETRIEVAL WITH VISION LANGUAGE MODELS
核心痛點：傳統系統在建構索引時，需要經過版面偵測、OCR 擷取與切塊等繁瑣步驟。不僅容易遺失圖表排版資訊，處理一頁甚至高達 7.22 秒。
突破性解法：
將文件視為圖像：ColPali 提出直接對「文件頁面的圖像」進行嵌入與檢索，完全跳過 OCR。
後期交互 (Late Interaction)：結合視覺語言模型 (PaliGemma-3B) 為圖像切塊生成多向量 (Multi-vector) 表示，並使用 ColBERT 風格的後期交互機制進行比對。
成效數據：
離線建立索引的時間從每頁 7.22 秒大幅縮短至 0.39 秒。
在富含圖表與複雜排版的文件中，檢索準確率顯著超越純文本檢索。


RAG-Anything & MMGraphRAG
Although images can be retrieved directly, complex questions often require "relational reasoning across different modalities." This is when scattered information needs to be "graphified." These two papers provide excellent structural solutions.
RAG-Anything (Macro Architecture): Introduce its Dual-Graph Construction strategy. Rather than forcing everything to merge, it simultaneously maintains a "text-based knowledge graph" and a "cross-modal knowledge graph." Through cross-modal hybrid retrieval, it combines structural navigation with semantic similarity matching, making it particularly suitable for processing ultra-long and highly heterogeneous documents.
MMGraphRAG (Micro Alignment): Traditional fusion methods lose fine-grained visual structures. The highlight of MMGraphRAG is treating visual nodes (like scene graphs) as first-class citizens in the graph. It innovatively uses SpecLink (based on spectral clustering) for highly accurate Cross-Modal Entity Linking (CMEL). This allows precise alignment between chart and text entities, preserving the complete reasoning path.
有了精準的檢索能力，但面對需要跨越不同模態（如文字敘述配合圖表數據）的複雜邏輯問題時，將資料打散成扁平的向量是不夠的，我們需要將其「結構化」。
論文參考 1：RAG-Anything: All-in-One RAG Framework
論文參考 2：MMGraphRAG: Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs
核心痛點：傳統的圖譜系統 (GraphRAG) 無法處理圖片；而傳統跨模態對齊（如圖片轉文字描述）又會遺失細粒度的結構化知識。
突破性解法：
雙圖建構策略 (Dual-Graph Construction) (RAG-Anything)：不強迫將所有內容壓扁，而是同時保留「純文本知識圖譜」與「跨模態知識圖譜」。
跨模態混合檢索 (RAG-Anything)：在檢索時，結合結構化知識的圖譜導航與密集向量的語義相似度匹配，在長篇且異質的文件中表現優異。
光譜聚類實體連結 (SpecLink) (MMGraphRAG)：將圖片提煉為場景圖，並把「視覺節點」視為圖譜中的一等公民。透過創新的 SpecLink 演算法，實現高準確度的跨模態實體連結 (CMEL)，讓圖表與文字實體精準對齊，保留完整的推理路徑。


Reasoning Optimization: Guiding VLMs to Search for Visual Evidence Like a Detective
VisRAG 2.0 (EVisRAG)
Narrative Logic: Is the task over once the retrieval system finds the correct images and feeds them to the generative model? No. Multiple images often cause VLMs to hallucinate or miss the point.
檢索系統找出多張圖片餵給生成模型後，任務還沒結束。模型面對多張檢索圖像時，經常會「看錯重點」或產生幻覺。
論文參考：VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation
核心痛點：在視覺 RAG 場景中，給予 VLM 越多檢索出來的圖片，模型越容易因為無法精準整合視覺證據而發生推理錯誤。
突破性解法 (EVisRAG)：
證據引導工作流：強制模型必須遵循像偵探一樣的明確步驟：觀察 (Observe) → 記錄證據 (Record Evidence) → 推理 (Reason) → 回答 (Answer)。
RS-GRPO 強化學習演算法：作者發明了「作用域限定的群組相對策略優化」(Reward-Scoped Group Relative Policy Optimization)。它將感知獎勵（找對圖片證據）與推導獎勵（答對問題）分別綁定在特定的 Token 範圍上，避免訓練時獎勵互相干擾。
成效數據：在多圖場景中精確定位相關證據，使問答準確率平均提升了 27%。