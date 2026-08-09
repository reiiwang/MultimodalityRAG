# Multimodal RAG 設計提案:Image / Table 檢索融合

## 1. 背景與問題

現有系統為純 text retrieval 的 RAG。為了支援多模態問答,需要新增 **image** 與 **table** 的檢索能力。

關鍵挑戰:圖片與表格的來源**不是單一管道**,而是兩種:

| 來源類型 | 說明 |
|---|---|
| 專屬檢索(`*_image` / `*_table`) | 由獨立的 image/table retriever(向量檢索)直接找出來的圖片、表格 |
| 附屬於文字(`text_image` / `text_table`) | 存在於被檢索到的 text chunk 內部,以 `<img>` html tag 呈現的圖片,或是內嵌在文字中、可能完整可能不完整的表格 |

因此每一種模態(image / table)在檢索完成後,都會有兩個候選池,需要設計融合(fusion)策略,再交給 VLM 產出對圖表的理解,最後拼接回 text chunk 作為 LLM 生成答案的加強 context。

## 2. 命名定義

- `image_image`:image retriever 找到的圖片
- `text_image`:text retriever 找到的 text chunk 中內嵌的圖片
- `table_table`:table retriever 找到的表格
- `text_table`:text retriever 找到的 text chunk 中內嵌的表格(完整或不完整皆可能)

## 3. 融合(Fusion)規則

### 3.1 Image(共 5 張)
| 來源 | 取數量 |
|---|---|
| `image_image` | Top-2(依 retrieval score 排序) |
| `text_image` | Top-3(依 retrieval score 排序) |

### 3.2 Table(共 3 個)
| 來源 | 取數量 |
|---|---|
| `table_table` | Top-1 |
| `text_table` | Top-2 |

### 3.3 動態配額與補位規則

Fusion 配額不是寫死的 2+3 / 1+2,而是「動態配額」:當某個來源候選數量不足時,由另一個來源依分數順序補滿。

**Image(目標 5 張)**
1. 先取 `text_image` Top-3、`image_image` Top-2
2. 若 `text_image` 候選不足 3 張(含完全沒有的情況),缺口由 `image_image` 依分數往下補,直到湊滿 5 張或雙邊來源都取盡

**Table(目標 3 個)**
1. 先取 `text_table` Top-2、`table_table` Top-1
2. 若 `text_table` 候選不足 2 個(含完全沒有的情況),缺口由 `table_table` 依分數往下補,直到湊滿 3 個或雙邊來源都取盡

> 反向情況(`image_image` / `table_table` 候選不足,由 `text_image` / `text_table` 補位)理論上也適用同一套邏輯,但實務上 image/table retriever 通常有足夠候選池,以 `text_image`/`text_table` 為主要補位方向即可,建議實作時仍寫成對稱的通用補位函式,不特別區分方向。

實作邏輯示意:
```
def fill_quota(primary_source, primary_n, secondary_source, secondary_n, total):
    picked = primary_source[:primary_n]
    remaining = total - len(picked)
    picked += secondary_source[:secondary_n]
    remaining = total - len(picked)
    if remaining > 0:
        # 用尚未被選中的 secondary_source 候選依分數補滿
        already_picked_ids = {p.id for p in picked}
        backup = [c for c in secondary_source[secondary_n:] if c.id not in already_picked_ids]
        picked += backup[:remaining]
    return picked[:total]

images = fill_quota(text_image, 3, image_image, 2, total=5)
tables = fill_quota(text_table, 2, table_table, 1, total=3)
```

## 4. VLM 處理階段

- 針對融合後的 5 張圖 + 3 個表,逐一組成 `(圖/表, user_query)` pair
- 呼叫 VLM,請其產出「該圖片/表格與 user query 的關聯性,或直接的答案描述」
- VLM 輸出為結構化文字(建議格式:簡短描述 + 與 query 的關聯 / 若可直接回答則給出答案片段)

## 5. 拼接策略

VLM 對每張圖/表產出描述後,依照來源類型決定拼接位置:

- **`text_image` / `text_table`**:天然有對應的 text chunk,VLM 的描述直接 append 到該 chunk 後面。
- **`image_image` / `table_table`(補位進來的、原本沒有對應 chunk)**:同樣 append 到**該圖片/表格本身所屬(出現)的 text chunk 後面**,但額外加上一段標註,說明這段描述是「由 image/table retrieval 額外補充,原文脈絡分數不高但圖表本身可能與 query 相關」。

  範例標註格式:
  ```
  [補充說明 - 此圖由 image retrieval 檢索取得,雖與所在文字段落的檢索分數不高,
  但圖片內容本身可能與問題相關]
  <VLM 產出的描述>
  ```

## 6. 去重與排序邏輯(重要)

同一張圖片/表格有可能同時被 `image_image`(或 `table_table`)檢索到,且該圖片/表格恰好也出現在某個 text chunk 裡(即也是 `text_image` / `text_table` 的候選)。此時採用以下規則:

1. **優先以「該圖片/表格所屬 text chunk 的檢索分數」排序、篩選**(而非以 image/table retriever 自身的相似度分數排序決定去留)。也就是說,若一張圖同時具備兩種身份,以它在 `text_image`/`text_table` 名單中的排序 & 是否入選為主。
2. 若該圖片所屬的 text chunk 分數不夠高、沒有被選進 `text_image` Top-3 / `text_table` Top-2,但這張圖仍在 `image_image` / `table_table` 的高分候選中,**依然視為獨立的 `image_image` / `table_table` 候選被選入**(不因為它「也出現在某個低分 chunk 裡」而被排除)。理由是:上下文分數低只代表這段文字跟 query 不太相關,不代表圖片/表格本身的內容跟 query 無關。
3. 避免同一張圖片/表格在最終 5 張(或 3 個)結果中重複出現兩次(例如同時被 `text_image` 和補位的 `image_image` 選中):以圖片/表格的唯一 ID 做去重,已選中的不再重複計入配額,並釋出名額給下一順位候選遞補。

## 7. 待確認 / 開放問題

1. **VLM 呼叫成本**:5 張圖 + 3 個表 = 最多 8 次 VLM 呼叫(或合併為 1 次多圖輸入),需評估延遲與成本,是否要對低分候選做 early cutoff。
2. **table 的「不完整表格」定義**:text chunk 中的不完整表格(例如被切斷的 markdown table)是否需要額外的修復/合併邏輯,或直接以殘缺形式送給 VLM。

## 8. 架構圖

見附件 `multimodal_rag_design.svg`


<svg viewBox="0 0 1180 900" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial, sans-serif">
  <rect width="1180" height="900" fill="#ffffff"/>

  <text x="590" y="40" text-anchor="middle" font-size="24" font-weight="bold" fill="#1a1a1a">Multimodal RAG 融合架構</text>

  <!-- Query -->
  <rect x="500" y="65" width="180" height="45" rx="8" fill="#2563eb"/>
  <text x="590" y="93" text-anchor="middle" font-size="15" fill="white" font-weight="bold">User Query</text>

  <!-- Arrows down to 3 retrievers -->
  <line x1="590" y1="110" x2="590" y2="140" stroke="#94a3b8" stroke-width="2"/>
  <line x1="590" y1="140" x2="200" y2="140" stroke="#94a3b8" stroke-width="2"/>
  <line x1="590" y1="140" x2="590" y2="140" stroke="#94a3b8" stroke-width="2"/>
  <line x1="590" y1="140" x2="980" y2="140" stroke="#94a3b8" stroke-width="2"/>
  <line x1="200" y1="140" x2="200" y2="165" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="590" y1="140" x2="590" y2="165" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="980" y1="140" x2="980" y2="165" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>

  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8"/>
    </marker>
  </defs>

  <!-- Text retriever -->
  <rect x="110" y="165" width="180" height="50" rx="8" fill="#16a34a"/>
  <text x="200" y="196" text-anchor="middle" font-size="14" fill="white" font-weight="bold">Text Retriever</text>

  <!-- Image retriever -->
  <rect x="500" y="165" width="180" height="50" rx="8" fill="#d97706"/>
  <text x="590" y="190" text-anchor="middle" font-size="14" fill="white" font-weight="bold">Image Retriever</text>
  <text x="590" y="207" text-anchor="middle" font-size="11" fill="#fff7ed">→ image_image</text>

  <!-- Table retriever -->
  <rect x="890" y="165" width="180" height="50" rx="8" fill="#7c3aed"/>
  <text x="980" y="190" text-anchor="middle" font-size="14" fill="white" font-weight="bold">Table Retriever</text>
  <text x="980" y="207" text-anchor="middle" font-size="11" fill="#f5f3ff">→ table_table</text>

  <!-- Text chunk box -->
  <rect x="70" y="245" width="260" height="90" rx="8" fill="#ecfdf5" stroke="#16a34a" stroke-width="2"/>
  <text x="200" y="270" text-anchor="middle" font-size="13" fill="#166534" font-weight="bold">Retrieved Text Chunks</text>
  <text x="200" y="292" text-anchor="middle" font-size="11" fill="#166534">內含 &lt;img&gt; tag</text>
  <text x="200" y="310" text-anchor="middle" font-size="11" fill="#166534">內含 完整/不完整 表格</text>

  <line x1="200" y1="215" x2="200" y2="245" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- extraction arrows -->
  <line x1="330" y1="270" x2="430" y2="270" stroke="#16a34a" stroke-width="2" stroke-dasharray="4" marker-end="url(#arrow2)"/>
  <line x1="330" y1="310" x2="430" y2="310" stroke="#16a34a" stroke-width="2" stroke-dasharray="4" marker-end="url(#arrow2)"/>
  <defs>
    <marker id="arrow2" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#16a34a"/>
    </marker>
  </defs>
  <text x="380" y="262" text-anchor="middle" font-size="10" fill="#166534">抽取</text>

  <!-- text_image / text_table boxes -->
  <rect x="430" y="245" width="150" height="40" rx="6" fill="#fef3c7" stroke="#d97706"/>
  <text x="505" y="270" text-anchor="middle" font-size="12" fill="#92400e" font-weight="bold">text_image</text>

  <rect x="430" y="295" width="150" height="40" rx="6" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="505" y="320" text-anchor="middle" font-size="12" fill="#5b21b6" font-weight="bold">text_table</text>

  <line x1="590" y1="215" x2="590" y2="245" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="500" y="245" width="0" height="0" fill="none"/>

  <line x1="980" y1="215" x2="980" y2="295" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- image_image box (from image retriever) -->
  <rect x="650" y="245" width="150" height="40" rx="6" fill="#fef3c7" stroke="#d97706"/>
  <text x="725" y="270" text-anchor="middle" font-size="12" fill="#92400e" font-weight="bold">image_image</text>

  <!-- table_table box (from table retriever) -->
  <rect x="820" y="295" width="150" height="40" rx="6" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="895" y="320" text-anchor="middle" font-size="12" fill="#5b21b6" font-weight="bold">table_table</text>

  <line x1="590" y1="215" x2="725" y2="245" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Merge to Image fusion pool -->
  <rect x="440" y="395" width="280" height="90" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="2"/>
  <text x="580" y="418" text-anchor="middle" font-size="14" font-weight="bold" fill="#92400e">Image 融合 Pool (共 5 張)</text>
  <text x="580" y="440" text-anchor="middle" font-size="12" fill="#92400e">image_image Top-2</text>
  <text x="580" y="458" text-anchor="middle" font-size="12" fill="#92400e">text_image Top-3</text>
  <text x="580" y="476" text-anchor="middle" font-size="10" fill="#b45309">(若無 text chunk → 5 張皆取自 image_image)</text>

  <line x1="505" y1="285" x2="500" y2="395" stroke="#d97706" stroke-width="2" marker-end="url(#arrow3)"/>
  <line x1="725" y1="285" x2="660" y2="395" stroke="#d97706" stroke-width="2" marker-end="url(#arrow3)"/>
  <defs>
    <marker id="arrow3" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#d97706"/>
    </marker>
  </defs>

  <!-- Merge to Table fusion pool -->
  <rect x="780" y="395" width="280" height="90" rx="8" fill="#f5f3ff" stroke="#7c3aed" stroke-width="2"/>
  <text x="920" y="418" text-anchor="middle" font-size="14" font-weight="bold" fill="#5b21b6">Table 融合 Pool (共 3 張)</text>
  <text x="920" y="440" text-anchor="middle" font-size="12" fill="#5b21b6">table_table Top-1</text>
  <text x="920" y="458" text-anchor="middle" font-size="12" fill="#5b21b6">text_table Top-2</text>
  <text x="920" y="476" text-anchor="middle" font-size="10" fill="#6d28d9">(若無 text chunk → 3 張皆取自 table_table)</text>

  <line x1="505" y1="335" x2="850" y2="395" stroke="#7c3aed" stroke-width="2" marker-end="url(#arrow4)"/>
  <line x1="895" y1="335" x2="920" y2="395" stroke="#7c3aed" stroke-width="2" marker-end="url(#arrow4)"/>
  <defs>
    <marker id="arrow4" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#7c3aed"/>
    </marker>
  </defs>

  <!-- VLM box -->
  <rect x="440" y="540" width="620" height="70" rx="8" fill="#2563eb"/>
  <text x="750" y="568" text-anchor="middle" font-size="16" font-weight="bold" fill="white">VLM (Vision-Language Model)</text>
  <text x="750" y="590" text-anchor="middle" font-size="12" fill="#dbeafe">輸入: (5 張圖 + 3 張表) × user query → 逐一產出關聯性 / 答案描述</text>

  <line x1="580" y1="485" x2="650" y2="540" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="920" y1="485" x2="850" y2="540" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Final merge back to text chunk -->
  <rect x="330" y="660" width="500" height="90" rx="8" fill="#ecfdf5" stroke="#16a34a" stroke-width="2"/>
  <text x="580" y="685" text-anchor="middle" font-size="15" font-weight="bold" fill="#166534">拼接回 Text Chunk</text>
  <text x="580" y="708" text-anchor="middle" font-size="12" fill="#166534">VLM 產出的圖/表描述 → append 至對應/最相關的 text chunk</text>
  <text x="580" y="726" text-anchor="middle" font-size="12" fill="#166534">形成增強後的 context</text>

  <line x1="750" y1="610" x2="580" y2="660" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- LLM answer -->
  <rect x="480" y="790" width="220" height="50" rx="8" fill="#1e293b"/>
  <text x="590" y="820" text-anchor="middle" font-size="14" fill="white" font-weight="bold">LLM 生成最終回答</text>
  <line x1="580" y1="750" x2="590" y2="790" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>

</svg>


<img width="197" height="150" alt="multimodal_rag_design" src="https://github.com/user-attachments/assets/e12aca21-a26e-4268-ad8f-c4af27ceb13d" />




---


<img width="170" height="150" alt="multimodal_rag_sequence_diagram" src="https://github.com/user-attachments/assets/4cbfcd03-5407-4c28-87b8-a5d951d422f0" />

<svg viewBox="0 0 1300 1150" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial, sans-serif">
  <rect width="1300" height="1150" fill="#ffffff"/>
  <text x="650" y="35" text-anchor="middle" font-size="22" font-weight="bold" fill="#1a1a1a">Multimodal RAG 循序圖 (Sequence Diagram)</text>

  <!-- Lifelines -->
  <!-- x positions -->
  <!-- User=80, Text=260, Image=440, Table=620, Fusion=820, VLM=1020, LLM=1200 -->
  <defs>
    <marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto">
      <path d="M0,0 L9,4 L0,8 Z" fill="#334155"/>
    </marker>
  </defs>

  <!-- headers -->
  <g font-size="13" font-weight="bold" fill="white">
    <rect x="30" y="55" width="100" height="36" rx="6" fill="#334155"/>
    <text x="80" y="78" text-anchor="middle">User</text>

    <rect x="200" y="55" width="120" height="36" rx="6" fill="#16a34a"/>
    <text x="260" y="78" text-anchor="middle">Text Retriever</text>

    <rect x="380" y="55" width="120" height="36" rx="6" fill="#d97706"/>
    <text x="440" y="78" text-anchor="middle">Image Retriever</text>

    <rect x="560" y="55" width="120" height="36" rx="6" fill="#7c3aed"/>
    <text x="620" y="78" text-anchor="middle">Table Retriever</text>

    <rect x="750" y="55" width="140" height="36" rx="6" fill="#0891b2"/>
    <text x="820" y="78" text-anchor="middle">Fusion Module</text>

    <rect x="960" y="55" width="120" height="36" rx="6" fill="#2563eb"/>
    <text x="1020" y="78" text-anchor="middle">VLM</text>

    <rect x="1150" y="55" width="100" height="36" rx="6" fill="#1e293b"/>
    <text x="1200" y="78" text-anchor="middle">LLM</text>
  </g>

  <!-- lifelines -->
  <g stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4,3">
    <line x1="80" y1="91" x2="80" y2="1110"/>
    <line x1="260" y1="91" x2="260" y2="1110"/>
    <line x1="440" y1="91" x2="440" y2="1110"/>
    <line x1="620" y1="91" x2="620" y2="1110"/>
    <line x1="820" y1="91" x2="820" y2="1110"/>
    <line x1="1020" y1="91" x2="1020" y2="1110"/>
    <line x1="1200" y1="91" x2="1200" y2="1110"/>
  </g>

  <!-- helper style for msg text -->
  <g font-size="12" fill="#1e293b">

  <!-- 1. User -> Text/Image/Table: query -->
  <line x1="80" y1="120" x2="255" y2="120" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>
  <text x="90" y="114" font-weight="bold">1: query</text>
  <line x1="80" y1="140" x2="435" y2="140" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>
  <text x="90" y="134" font-weight="bold">1: query</text>
  <line x1="80" y1="160" x2="615" y2="160" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>
  <text x="90" y="154" font-weight="bold">1: query</text>

  <!-- activation bars -->
  <rect x="255" y="120" width="10" height="60" fill="#bbf7d0"/>
  <rect x="435" y="140" width="10" height="40" fill="#fde68a"/>
  <rect x="615" y="160" width="10" height="20" fill="#ddd6fe"/>

  <!-- 2. Text Retriever returns text chunks -->
  <line x1="260" y1="200" x2="825" y2="200" stroke="#16a34a" stroke-width="1.5" marker-end="url(#a)"/>
  <text x="270" y="194" fill="#166534" font-weight="bold">2: text_chunks (含 &lt;img&gt; / 表格)</text>

  <!-- 3. Image retriever returns image_image -->
  <line x1="440" y1="220" x2="825" y2="220" stroke="#d97706" stroke-width="1.5" marker-end="url(#a)"/>
  <text x="450" y="214" fill="#92400e" font-weight="bold">3: image_image candidates</text>

  <!-- 4. Table retriever returns table_table -->
  <line x1="620" y1="240" x2="825" y2="240" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#a)"/>
  <text x="630" y="234" fill="#5b21b6" font-weight="bold">4: table_table candidates</text>

  <!-- Fusion module activation -->
  <rect x="815" y="200" width="10" height="330" fill="#a5f3fc"/>

  <!-- 5. Fusion: extract text_image / text_table -->
  <rect x="700" y="260" width="240" height="40" rx="6" fill="#ecfeff" stroke="#0891b2"/>
  <text x="820" y="277" text-anchor="middle" fill="#0e7490" font-size="11" font-weight="bold">5: 從 text_chunks 抽取</text>
  <text x="820" y="292" text-anchor="middle" fill="#0e7490" font-size="11" font-weight="bold">text_image / text_table</text>

  <!-- 6. Fusion decision box -->
  <rect x="640" y="315" width="360" height="90" rx="6" fill="#fff" stroke="#0891b2" stroke-width="1.5" stroke-dasharray="3,3"/>
  <text x="820" y="332" text-anchor="middle" fill="#0e7490" font-size="12" font-weight="bold">6: alt 動態配額 fusion</text>
  <text x="655" y="350" font-size="11" fill="#164e63">[有 text chunk 命中]</text>
  <text x="670" y="365" font-size="11" fill="#164e63">image = text_image(≤3) + image_image 補滿至 5</text>
  <text x="670" y="379" font-size="11" fill="#164e63">table = text_table(≤2) + table_table 補滿至 3</text>
  <line x1="640" y1="386" x2="1000" y2="386" stroke="#0891b2" stroke-dasharray="2,2"/>
  <text x="655" y="399" font-size="11" fill="#164e63">[無 text chunk 命中] image=image_image Top5 / table=table_table Top3</text>

  <!-- 7. dedup -->
  <rect x="700" y="415" width="240" height="34" rx="6" fill="#ecfeff" stroke="#0891b2"/>
  <text x="820" y="436" text-anchor="middle" fill="#0e7490" font-size="11" font-weight="bold">7: 依唯一 ID 去重 + 依 chunk 分數排序</text>

  <!-- 8. Fusion -> VLM: send 5 images + 3 tables -->
  <line x1="825" y1="460" x2="1015" y2="460" stroke="#0891b2" stroke-width="1.5" marker-end="url(#a)"/>
  <text x="835" y="454" fill="#0e7490" font-weight="bold">8: (5 images + 3 tables, query)</text>

  <rect x="1015" y="460" width="10" height="80" fill="#bfdbfe"/>

  <!-- 9. VLM loop -->
  <rect x="960" y="480" width="200" height="60" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5" stroke-dasharray="3,3"/>
  <text x="1060" y="497" text-anchor="middle" fill="#1d4ed8" font-size="11" font-weight="bold">9: loop 每張圖/表</text>
  <text x="1060" y="512" text-anchor="middle" fill="#1d4ed8" font-size="11">VLM(image/table, query)</text>
  <text x="1060" y="526" text-anchor="middle" fill="#1d4ed8" font-size="11">→ 關聯性描述 / 答案</text>

  <!-- 10. VLM -> Fusion: descriptions -->
  <line x1="1020" y1="560" x2="825" y2="560" stroke="#2563eb" stroke-width="1.5" marker-end="url(#a)"/>
  <text x="835" y="554" fill="#1d4ed8" font-weight="bold">10: 回傳 8 段描述文字</text>

  <!-- 11. Fusion: splice back -->
  <rect x="640" y="580" width="360" height="70" rx="6" fill="#ecfeff" stroke="#0891b2" stroke-width="1.5"/>
  <text x="820" y="598" text-anchor="middle" fill="#0e7490" font-size="12" font-weight="bold">11: 拼接回對應 text chunk</text>
  <text x="660" y="616" font-size="10.5" fill="#164e63">text_image/text_table → append 至原 chunk</text>
  <text x="660" y="632" font-size="10.5" fill="#164e63">image_image/table_table(補位)→ append 至所屬 chunk</text>
  <text x="660" y="645" font-size="10.5" fill="#164e63">+ 標註「由 retrieval 額外補充」</text>

  <!-- 12. Fusion -> LLM -->
  <line x1="825" y1="670" x2="1195" y2="670" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>
  <text x="835" y="664" font-weight="bold">12: 增強後 context</text>

  <rect x="1195" y="670" width="10" height="50" fill="#94a3b8"/>

  <rect x="1050" y="690" width="140" height="34" rx="6" fill="#1e293b"/>
  <text x="1120" y="711" text-anchor="middle" fill="white" font-size="11" font-weight="bold">13: 生成回答</text>

  <!-- 14. LLM -> User -->
  <line x1="1200" y1="740" x2="85" y2="740" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>
  <text x="900" y="734" font-weight="bold">14: final answer</text>

  </g>

  <!-- footer note -->
  <rect x="30" y="790" width="1240" height="120" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="50" y="815" font-size="13" font-weight="bold" fill="#1a1a1a">補充說明</text>
  <text x="50" y="838" font-size="12" fill="#334155">• step 6 的 alt 分支即為動態配額補位邏輯:text_image/text_table 不足時,由 image_image/table_table 依分數往下補。</text>
  <text x="50" y="858" font-size="12" fill="#334155">• step 7 去重規則:同一張圖/表若同時出現在兩種來源,優先依其所屬 text chunk 分數決定去留;若所屬 chunk 分數低但圖表本身仍在 image_image/table_table 高分候選中,仍視為獨立候選入選。</text>
  <text x="50" y="878" font-size="12" fill="#334155">• step 9 的 VLM 呼叫可視實作改為單次多圖輸入,而非逐一 loop,以降低延遲。</text>

</svg>












