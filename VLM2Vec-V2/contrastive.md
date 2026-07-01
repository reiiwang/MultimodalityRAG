# 3.2 對比學習(Contrastive Learning)理論解釋

## 核心思路:如何把「生成模型」變成「嵌入模型」

Qwen2-VL 這類 VLM 本質上是生成式模型(預測下一個 token),但我們要的是能輸出「一個向量代表整個輸入語意」的嵌入模型。這段理論說明的就是**怎麼把生成模型改造/微調成嵌入模型**——答案是靠對比學習(contrastive learning)。

---

## 第一步:如何從 VLM 得到一個「向量」

給定一個 query(例如:指令+圖片+文字)和一個 target(例如:候選的圖片或文字),分別餵進 VLM,然後**取最後一層、最後一個 token 的向量表示**,當作這段輸入的嵌入:

$$h_{q}^{inst}, \quad h_{t^+}$$

- $h_{q}^{inst}$:query 的嵌入(上標 inst 表示這個 query 帶有指令 instruction)
- $h_{t^+}$:正確答案(positive target)的嵌入

**為什麼取最後一個 token?**
因為 Qwen2-VL 是自迴歸(causal attention)架構,最後一個 token 在注意力機制下已經「看過」前面所有的 token,理論上它的向量已經融合了整段輸入的完整語意資訊,所以可以拿它當作整體輸入的代表向量(pooling 的一種簡單做法)。

---

## 第二步:訓練目標——InfoNCE Loss

現在我們有了 query 向量和 target 向量,接下來要**訓練模型讓正確配對的向量靠近、錯誤配對的向量遠離**。這正是對比學習的核心精神。

論文採用標準的 InfoNCE loss:

$$
\mathcal{L} = -\log \frac{\varphi(h_q^{inst}, h_{t^+})}{\varphi(h_q^{inst}, h_{t^+}) + \sum_{t^- \in \mathcal{N}} \varphi(h_q^{inst}, h_{t^-})}
$$

我們可以把它拆解成幾個部分來理解:

### (1) 分子:正樣本的匹配分數
$\varphi(h_q^{inst}, h_{t^+})$ 代表 query 和「正確答案」之間的相似度分數。我們希望這個值**越大越好**。

### (2) 分母:正樣本 + 所有負樣本的分數總和
$\mathcal{N}$ 是負樣本集合,包含兩種來源:
- **In-batch negatives(批次內負樣本)**:同一個訓練 batch 裡,其他樣本的 target 就自動被當作這個 query 的負樣本。這是一種很高效的做法——不需要額外挑選負樣本,batch size 越大,負樣本就越多,訓練訊號也越豐富。
- **Hard negatives(困難負樣本)**:刻意挑選出「看起來很像正確答案,但其實是錯的」樣本。這種負樣本比隨機負樣本更能逼迫模型學到細緻的語意區分能力(而不是只學會表面特徵)。

### (3) 整體形式:一個 softmax 分類問題
這個 loss 本質上等同於:**把「找出正確 target」當成一個多分類問題**,分類的類別數就是「1 個正樣本 + 所有負樣本」的數量,而 loss 就是這個分類問題的 cross-entropy。

$$
\mathcal{L} = -\log(\text{softmax 後,正確答案所佔的機率})
$$

最小化這個 loss,就是在訓練模型:讓正確配對的相似度分數,在所有候選(正確+錯誤)中**佔比越高越好**。

---

## 第三步:相似度函數 φ 的具體形式

$$
\varphi(h_q, h_t) = \exp\left(\frac{1}{\tau}\cos(h_q, h_t)\right)
$$

這裡有兩個關鍵設計:

### (1) 餘弦相似度(cosine similarity)
$\cos(h_q, h_t)$ 衡量兩個向量的**方向**是否一致,而不管向量的長度(norm)。這在嵌入學習中很常見,因為我們通常只關心語意方向是否一致,而不是向量的絕對大小。

### (2) 溫度參數 τ(temperature)
$\tau$ 放在 cosine similarity 前面,作用是**縮放相似度分數的分佈**:
- $\tau$ 越小 → 把相似度差異放大 → softmax 分佈更「尖銳」(sharp),模型對負樣本的懲罰更嚴厲,訓練訊號更強但可能不穩定
- $\tau$ 越大 → 相似度差異被壓縮 → softmax 分佈更「平滑」,訓練較穩定但區分力較弱

$\tau$ 通常是一個需要調的超參數(或設為可學習參數),用來控制訓練難度與收斂穩定性之間的平衡。

---

## 整體串起來看

這整套機制在做的事情是:

1. 把 query(帶指令)和 target 都編碼成向量
2. 用 cosine similarity + 溫度縮放,計算「query 與各個候選 target」的匹配分數
3. 用 InfoNCE(本質上是 softmax cross-entropy)訓練模型,讓正確配對的分數在所有候選中最高
4. 訓練過程中同時利用 in-batch negatives(免費、大量)和 hard negatives(精挑細選、更有效)來提供區分訊號

這正好呼應你前面問的問題:Qwen2-VL 只提供了「編碼」的能力,而**這一節的對比學習,就是讓嵌入空間真正具備「語意相近則距離近」這個性質的關鍵訓練機制**——這是架構本身不會自動產生的能力,必須靠這個 loss 函數在訓練過程中「雕塑」出來。

如果你想,我可以進一步畫一張圖說明整個 pipeline(query/target 輸入 → VLM 編碼 → pooling → 相似度計算 → loss),會更直觀。


---

Q: 所以 帶指令的 query+target 這樣一個 concat input 喂進 Qwen2-VL 之後 他去取最後一層的向量來表示這個 concat input 然後哪裡來的 各個候選 target？


好問題,這裡有個關鍵誤解需要先釐清:**query 和 target 不是 concat 在一起餵進去的,而是分開、各自獨立跑一次 forward pass。**

## 澄清:是「雙塔式」(dual-encoder / two-tower),不是「拼接式」

論文原文說的是:

> we feed **query and target** into it to obtain the **query and target embeddings** $(h_q^{inst}, h_{t^+})$

注意這裡是**兩個獨立的向量**,不是一個。實際流程是:

```
第一次 forward:
輸入 = [指令 + query 內容(文字/圖片)]  →  Qwen2-VL  →  取最後一個 token 向量  →  h_q^inst

第二次 forward:
輸入 = [target 內容(文字/圖片)]        →  Qwen2-VL  →  取最後一個 token 向量  →  h_t
```

這就是所謂的 **dual-encoder 架構**(也叫 two-tower,雖然這裡是同一個模型權重共用,而不是真的兩個獨立模型)。query 和 target 各自產生自己的一個向量,兩者事後才用 cosine similarity 去比對。

**為什麼不能 concat 在一起?**
如果把 query 和 target 拼在一起餵進去,那 attention 機制會讓兩者的資訊直接互相「看到、混合」,最後只會得到「這對 query-target 匹不匹配」的單一分數(這其實就是 cross-encoder 的做法,常見於 reranker,而不是 embedding model)。但這樣的話:
- 沒辦法預先把資料庫裡幾百萬個 target 都編碼好存起來(因為每次都要跟新的 query 重新一起跑一次)
- 沒辦法做快速的向量檢索(ANN search)

而 embedding model 的目的正是要讓 target 可以**獨立於任何 query,預先編碼好存成向量資料庫**,之後 query 來了只需要算 cosine similarity 就好,速度極快。

---

## 那「各個候選 target」到底哪裡來的?

現在回到你的問題核心。在一個訓練 batch 裡,假設 batch size = 4,你會有 4 組 (query, target) pair:

| | Query | Positive Target |
|---|---|---|
| 樣本1 | q1(指令+圖) | t1 |
| 樣本2 | q2 | t2 |
| 樣本3 | q3 | t3 |
| 樣本4 | q4 | t4 |

**In-batch negatives 的做法是:**

對於 q1 來說:
- 正樣本 = t1(它自己對應的答案)
- 負樣本 = t2, t3, t4(**別人的答案**,直接拿來當作 q1 的錯誤候選)

也就是說,不需要額外標記「這是負樣本」,只要把同一個 batch 裡**其他樣本的 target**都當作負樣本用就好——這是完全免費、自動產生的負樣本來源。batch size 越大,每個 query 能看到的負樣本就越多。

**Hard negatives 的做法是:**

額外再塞一些「刻意挑選、容易混淆」的 target 進同一個 batch(例如用某種相似度檢索先找出「長得很像正確答案但其實是錯的」樣本),讓 $\mathcal{N}$ 這個負樣本集合裡不只有隨機的 in-batch 負樣本,還有更難分辨的困難負樣本。這些通常是資料前處理階段就先準備好、跟著這筆訓練資料一起打包進 batch 的。

---

## 整體流程圖(文字版)

```
Batch 內有 B 筆 (query, positive target) pair

Step 1: 把所有 query 分別餵進 Qwen2-VL → 得到 B 個 h_q 向量
Step 2: 把所有 target(正樣本 + hard negatives)分別餵進 Qwen2-VL → 得到 (B + 額外negative數) 個 h_t 向量
Step 3: 計算每個 h_q 跟「所有 h_t」的 cosine similarity → 形成一個相似度矩陣
Step 4: 對每一個 query,它對應的正確 target 應該是那一列裡分數最高的
         → 用 InfoNCE(等於 softmax cross-entropy)訓練
```

所以本質上這是一個 **B × (B + extra negatives) 的相似度矩陣**,對角線(或標記出的正確位置)是正樣本分數,其餘位置全部當負樣本,拿去算 loss。

這也解釋了為什麼大 batch size 對 contrastive learning 特別重要——batch 越大,矩陣越大,每個 query 免費得到的負樣本就越多,訓練訊號越豐富。

---

Q: Step 3: 計算每個 h_q 跟「所有 h_t」的 cosine similarity → 形成一個相似度矩陣 Step 4: 對每一個 query,它對應的正確 target 應該是那一列裡分數最高的 這裡是什麼意思 具體怎麼訓練

好,我用一個具體的數值例子把整個訓練過程攤開講。

## 情境設定

假設 batch size = 4,經過 Step 1、2,你已經拿到:

- 4 個 query 向量:$h_{q1}, h_{q2}, h_{q3}, h_{q4}$
- 4 個 target 向量(先不考慮 hard negatives):$h_{t1}, h_{t2}, h_{t3}, h_{t4}$

其中配對關係是:q1↔t1、q2↔t2、q3↔t3、q4↔t4(這是資料集本身給定的正確答案)。

---

## Step 3:算出相似度矩陣

把每個 query 向量跟每個 target 向量都算一次 cosine similarity(再除以溫度 τ、取 exp,或者先算 logits 再說),會得到一個 **4×4 的矩陣**:

|        | t1   | t2   | t3   | t4   |
|--------|------|------|------|------|
| **q1** | 8.2  | 1.1  | 0.5  | 2.0  |
| **q2** | 0.9  | 7.8  | 1.5  | 0.3  |
| **q3** | 1.2  | 0.7  | 6.9  | 1.8  |
| **q4** | 0.4  | 1.0  | 2.1  | 8.5  |

(這些數字是 $\cos(h_q, h_t)/\tau$ 算出來的 logits,只是示意)

矩陣裡每一格代表「這個 query 跟這個 target 有多相似」。

**對角線**(q1-t1, q2-t2, q3-t3, q4-t4)是**正確配對**,理想情況下應該是每一列裡數值最大的位置。你可以看到在這個例子裡,對角線的值(8.2, 7.8, 6.9, 8.5)確實都比同一列的其他值大——這代表模型現在已經學得不錯了。但訓練初期,這些數字通常是雜亂的,對角線不見得最大。

---

## Step 4:每一列當成一個分類問題

**關鍵理解:把矩陣的「每一列」當作一個獨立的分類任務。**

以 **q1 這一列**為例:[8.2, 1.1, 0.5, 2.0]

這四個數字代表「q1 分別和 t1, t2, t3, t4 的相似度」。我們把這當成一個 4 分類問題——**q1 到底該配對哪一個 target?正確答案是 index 0(也就是 t1)**。

做法就是標準的 softmax + cross-entropy:

**(1) 先做 softmax,把相似度轉成機率分佈**

$$p(t_i | q1) = \frac{e^{8.2}}{e^{8.2}+e^{1.1}+e^{0.5}+e^{2.0}}$$

算出來大概會是:
- p(t1|q1) ≈ 0.994
- p(t2|q1) ≈ 0.002
- p(t3|q1) ≈ 0.001
- p(t4|q1) ≈ 0.005

這組機率的意思是:「模型認為 q1 有 99.4% 的機率應該配對 t1」。

**(2) 算 cross-entropy loss**

因為真實標籤是 t1(one-hot: [1,0,0,0]),cross-entropy loss 就是:

$$\mathcal{L}_{q1} = -\log(p(t1|q1)) = -\log(0.994) \approx 0.006$$

**這個值很小,代表模型在這個樣本上表現很好(幾乎確定答案是 t1)。**

如果模型還沒訓練好,假設 logits 是 [1.0, 3.5, 0.8, 1.2](正確答案 t1 反而分數不高),那麼:
- p(t1|q1) 可能只有 0.05
- $\mathcal{L}_{q1} = -\log(0.05) \approx 3.0$(loss 很大,代表模型錯得離譜)

**這正是論文公式 (1) 在做的事情:**

$$\mathcal{L} = -\log \frac{\varphi(h_q, h_{t^+})}{\varphi(h_q, h_{t^+}) + \sum_{t^-} \varphi(h_q, h_{t^-})}$$

分子是正確答案的 exp(相似度),分母是「正確答案 + 所有負樣本」的 exp(相似度)總和——這其實就是 softmax 的定義,取 log 再加負號就是 cross-entropy。

**(3) 對每一列都重複這個計算**

q2 這一列 [0.9, 7.8, 1.5, 0.3],正確答案是 index 1(t2) → 算一次 loss
q3 這一列,正確答案是 index 2(t3) → 算一次 loss
q4 這一列,正確答案是 index 3(t4) → 算一次 loss

**(4) 整個 batch 的總 loss = 4 個 loss 的平均**

$$\mathcal{L}_{batch} = \frac{1}{4}(\mathcal{L}_{q1} + \mathcal{L}_{q2} + \mathcal{L}_{q3} + \mathcal{L}_{q4})$$

---

## 反向傳播:模型實際上學到什麼?

拿到這個 scalar loss 之後,做標準的 backpropagation,梯度會往回傳,調整 Qwen2-VL 的參數(通常是用 LoRA 或全參數微調)。

直觀上,梯度會推動模型做兩件事:

1. **拉近正確配對**:讓 $\cos(h_{q1}, h_{t1})$ 變大 → 調整編碼方式,讓 q1 和 t1 的向量方向更接近
2. **推遠錯誤配對**:讓 $\cos(h_{q1}, h_{t2})$, $\cos(h_{q1}, h_{t3})$, $\cos(h_{q1}, h_{t4})$ 變小 → 讓 q1 跟其他不相關的 target 向量方向更遠離

這個「拉近正確、推遠錯誤」的操作,會**同時對這個 batch 裡所有 4 個 query 都做一次**,因為 t1 不只是 q1 的正樣本,它同時也是 q2、q3、q4 的負樣本(對它們而言)。所以一次 batch 的訓練,其實同時在優化「4×4 = 16 個 pair 的相對距離關係」,而不是只有 4 個。

---

## 加上 Hard Negatives 後的差異

如果每個 query 還額外配有 2 個 hard negatives,矩陣就會變成 **4 列 × (4+8) 欄**(4 個 in-batch targets + 4×2 個額外 hard negatives),分母的求和項變多,但邏輯完全一樣——只是每一列的「候選答案」數量變多了,分類問題從 4 選 1 變成 12 選 1,訓練訊號更精細。

---

## 一句話總結整個機制

**這其實就是一個「用相似度矩陣做多分類」的訓練方式**:每個 query 是一筆分類任務,candidates 就是 batch 裡所有的 target(正確的 + 別人的當負的 + 額外準備的 hard negatives),用 cross-entropy 逼模型把正確答案的相似度分數拉到最高、其他全部壓低,如此反覆訓練,最終讓整個嵌入空間具備「語意相近就距離近」的性質。