# Shop-the-Look: Deep Multimodal Visual Product Discovery 🛍️✨

An end-to-end Machine Learning visual search and recommendation engine designed to bridge the **"Street-to-Shop" domain gap**. Given an in-the-wild scene photograph of an outfit, the system automatically detects individual garments, retrieves matching pieces from a **28,094-product catalog**, determines exact match candidates vs. fallback recommendations, and provides multi-attribute explainability (color, category, silhouette, texture).

---

## 1. System Architecture

```text
                  Scene Photo (Unconstrained Query)
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │    Stage 1: Multi-Garment Parsing   │
              │   Grounding DINO (Zero-Shot ViT)    │
              │  + Non-Maximum Suppression (IoU=0.45)│
              └─────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
        Query Crop #1 (Shoes)            Query Crop #2 (Denim)
                │                                 │
                ▼                                 ▼
              ┌─────────────────────────────────────┐
              │    Stage 2: Metric Representation   │
              │   DINOv2 ViT-S/14 (L2-Normalized)   │
              │    + Aspect-Ratio Square Padding    │
              └─────────────────────────────────────┘
                                 │
                                 ▼   q ∈ R^384
              ┌─────────────────────────────────────┐
              │   Stage 3: Sub-2ms Vector Search    │
              │  Catalog Dot-Product / FAISS Matrix │
              │   + Category Isolation (732 Glasses)│
              └─────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
        Score s ≥ τ (Provisional)         Score s < τ (Provisional)
        "MATCH CANDIDATE"                 "STYLE RECOMMENDATIONS"
                │                                 │
                └────────────────┬────────────────┘
                                 ▼
              ┌─────────────────────────────────────┐
              │  Stage 4: Multi-Attribute Explainer │
              │  CIELAB ΔE, Texture Energy, Framing │
              └─────────────────────────────────────┘
```

---

## 2. Repository Structure

```text
OC/
├── app.py                      # Interactive Streamlit Web Application (Deliverable 3)
├── pipeline.py                 # Core unified inference engine (Grounding DINO + DINOv2 + FAISS)
├── explainability.py           # Multi-attribute attribution engine (CIELAB ΔE, texture, silhouette)
├── calibrate_validation.py     # Evaluation & decision boundary threshold calibration script
├── build_catalog_index.py      # Offline indexing pipeline (extracts 28k DINOv2 packshot embeddings)
├── TECHNICAL_REPORT.md         # Comprehensive 5-Page Technical Report (Deliverable 2)
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation & execution guide
├── .gitignore                  # Git ignore rules for datasets, venv, and cache
│
├── dataset/                    # Ground-truth dataset metadata
│   ├── product_catelog.jsonl   # 50,000 product impression records (28,094 unique signatures)
│   └── validation.jsonl        # 330 labeled pairs (scene signature → product signature)
│
├── index/                      # Precomputed index assets
│   ├── catalog_matrix.npy      # Normalized float32 embedding matrix (28094 x 384)
│   ├── catalog_sigs.json       # Ordered mapping of vector row IDs to product signatures
│   ├── catalog_aspect_ratios.npy # Precomputed aspect ratios for category filtering
│   ├── catalog_is_glasses.npy  # Boolean mask identifying 732 verified eyewear packshots
│   └── catalog.faiss           # FAISS IndexFlatIP binary vector index
│
├── results/
│   └── calibration_report.json # Quantitative evaluation & threshold sweep results
│
├── data/                       # (Ignored by Git) Downloaded image assets (~605 MB)
│   ├── scenes/                 # 321 downloaded validation scene photos
│   └── products/               # 28,135 downloaded catalog packshot photos
│
└── venv/                       # (Ignored by Git) Python virtual environment
```

---

## 3. Quickstart & Installation

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone <your-repo-url>
cd OC

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Launch the Interactive Streamlit Demo (Deliverable 3)
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.
* **Pre-loaded Samples:** Select sample street looks from the validation set (Look #1 Sneakers, Look #2 Summer Blouse, Look #4 Ripped Jeans).
* **Upload Image:** Upload any photo or selfie to test real-world outfit detection, catalog retrieval, and multi-attribute explanation.
* **Provisional Threshold Slider:** Experiment with threshold $\tau$ to dynamically inspect the exact-match decision boundary.

### Step 3: Run Quantitative Validation & Calibration Benchmark
```bash
python3 calibrate_validation.py
```
This script benchmarks the pipeline against the ground-truth labeled validation set, evaluating Top-$K$ Recall, MRR, positive vs. negative score distributions, and saving results to `results/calibration_report.json`.

---

## 4. Benchmark & Evaluation Results

Tested across ground-truth validation pairs linking real-world street scenes to catalog packshots:

| Metric | Full Scene Naive Baseline | Grounding DINO + DINOv2 (Ours) | Improvement |
| :--- | :---: | :---: | :---: |
| **Recall@1** | 0.35% | **18.75%** | **~50x Gain** |
| **Recall@5** | 0.69% | **37.50%** | **~54x Gain** |
| **Recall@10** | 1.04% | **46.88%** | **~45x Gain** |
| **MRR** | 0.0051 | **0.2642** | **~52x Gain** |
| **Latency** | 2.1 ms | **1.85 ms (vector search)** | Real-Time |

### Score Distribution & Threshold Calibration ($\tau$)

| Population | Sample Count | Mean Cosine | Median | Range |
| :--- | :---: | :---: | :---: | :---: |
| **Random Catalog Items** | 500 | 0.0681 | 0.0439 | $[-0.075, 0.604]$ |
| **Positive Matches (Crop vs. Packshot)** | 40 | **0.4622** | **0.4836** | $[0.100, 0.771]$ |
| **Out-of-Catalog Top-1 Distractors** | 212 | **0.5865** | **0.5918** | $[0.269, 0.907]$ |

> **Key Finding:** In a 28,094-product catalog, nearest-neighbor search for out-of-catalog items yields an average score of $\sim 0.58$. Therefore, static thresholds (e.g. $\tau = 0.226$) are designated as **provisional exact-match candidates**, requiring multi-attribute validation or secondary verification to confirm exact SKU identity.

---

## 5. Edge-Case Engineering Highlights

1. **Aspect-Ratio Preserving Square Padding:** Prevents vertical squashing of low-top sneakers and horizontal items, improving ground-truth retrieval rank by over 25x.
2. **IoU Non-Maximum Suppression (NMS):** Deduplicates candidate boxes ($\theta_{\text{IoU}} = 0.45$) to eliminate duplicate detections for the same physical garment.
3. **Verified Eyewear Sub-Index:** Dedicated zero-shot classifier tagged all 732 glasses in the catalog (`catalog_is_glasses.npy`), preventing footwear and handbags from leaking into eyewear searches.
4. **Honest, Confidence-Aware Explainability:** Dynamically identifies when an item is a low-confidence fallback ($s < 0.45$), eliminating hallucinated attribute claims and building user trust.

---

## 6. Deliverable Checklist

- [x] **Deliverable 1: Source Code**: Modular, documented code (`pipeline.py`, `explainability.py`, `app.py`).
- [x] **Deliverable 2: Technical Report**: 5-page structured scientific document (`TECHNICAL_REPORT.md`).
- [x] **Deliverable 3: Interactive Demo**: Streamlit web UI supporting sample inspection, custom uploads, bounding boxes, recommendations, and attribution cards.
