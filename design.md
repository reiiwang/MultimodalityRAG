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
