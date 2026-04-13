# 🛒 RAG Product Manager

A local AI-powered inventory tracking agent built with **LlamaIndex**, **Ollama**, and **Retrieval-Augmented Generation (RAG)**. Ask natural-language questions about your product catalog and get structured, actionable inventory reports — all running 100% offline on your machine.

---

## 📋 Table of Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [CSV Format](#csv-format)
- [Running the Agent](#running-the-agent)
- [Example Output](#example-output)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

---

## What It Does

The **RAG Product Manager** loads a product catalog from a CSV file, embeds each product into a vector index, and lets you query your inventory using natural language. The agent automatically:

- ✅ Flags products that are **out of stock** (`stock_quantity = 0`)
- ⚠️ Identifies products that **need reordering** (`stock_quantity < reorder_point`)
- 🏆 Highlights **top performers** (`rating ≥ 4.6` AND `last_month_sales > 200`)
- 🚨 Detects **supply risks** (`lead_time_days ≥ 7` AND `stock_quantity < reorder_point`)
- 📋 Ends every report with a **Priority Actions** section

---

## How It Works

```
products.csv
     │
     ▼
Build TextNodes          ← Each product becomes a node with text + metadata
     │
     ▼
OllamaEmbedding          ← nomic-embed-text embeds the text into vectors
     │
     ▼
VectorStoreIndex         ← In-memory vector store (LlamaIndex)
     │
     ▼
Query Engine             ← Retrieves top-k nodes, injects them into prompt
     │
     ▼
Ollama LLM (gemma4:e2b)  ← Generates structured inventory report
     │
     ▼
Printed Report
```

The key design choice is **metadata separation**: the embedder only sees product name, category, supplier, price, and rating (semantic info), while the LLM sees the full operational metadata (stock levels, reorder points, lead times, sales) needed to write accurate reports.

---

## Prerequisites

Make sure the following are installed on your machine before proceeding.

### 1. Python 3.10+

Download from [python.org](https://www.python.org/downloads/).

Verify:
```bash
python --version
```

### 2. Ollama

Download and install from [ollama.com](https://ollama.com/download).

Verify:
```bash
ollama --version
```

### 3. Required Ollama Models

Pull the two models the project uses:

```bash
# The LLM (language model for generating reports)
ollama pull gemma4:e2b

# The embedding model (for vector search)
ollama pull nomic-embed-text
```

This may take a few minutes depending on your connection. You only need to do this once.

Verify both are available:
```bash
ollama list
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Abdou-Moumen/RAG-Prodect-Manager.git
cd RAG-Prodect-Manager
```

### 2. (Recommended) Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama
```

Or if a `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
RAG-Prodect-Manager/
│
├── Reporter.py          # Main script — builds index and runs the agent
├── products.csv         # Product catalog (your data source)
└── README.md            # This file
```

---

## CSV Format

Your `products.csv` must have the following columns (order does not matter):

| Column | Type | Description |
|---|---|---|
| `product_id` | string | Unique identifier, e.g. `P001` |
| `name` | string | Product display name |
| `category` | string | Product category |
| `supplier` | string | Supplier name |
| `price` | float | Unit price in USD |
| `stock_quantity` | int | Current units in stock |
| `reorder_point` | int | Minimum stock before reorder |
| `last_month_sales` | int | Units sold last month |
| `rating` | float | Customer rating out of 5 |
| `lead_time_days` | int | Days to receive new stock from supplier |

### Example row

```csv
product_id,name,category,supplier,price,stock_quantity,reorder_point,last_month_sales,rating,lead_time_days
P001,UltraBook Pro Laptop,Electronics,TechSupply Inc,1299.99,47,20,156,4.8,5
```

---

## Running the Agent

Make sure Ollama is running in the background (it starts automatically on most systems, or run `ollama serve`), then execute:

```bash
python Reporter.py
```

### What happens step by step

1. **Model warm-up** — sends a quick `"hi"` to Ollama to load `gemma4:e2b` into memory before the real query
2. **Node building** — reads `products.csv` and creates 15 TextNodes with separated embed/LLM metadata
3. **Index inspection** — prints a detailed view of the first product node (embed view, LLM view, raw metadata)
4. **Vector indexing** — embeds all nodes using `nomic-embed-text` (progress bar shown)
5. **Query** — asks the agent *"Which products are performing the best?"* and prints the structured report

---

## Example Output

```
Warming up model...
Model warm
Models configured
   LLM      : gemma4:e2b
   Embedder : nomic-embed-text
Built 15 nodes

[Node inspection for UltraBook Pro Laptop...]

Building vector index...
Generating embeddings: 100%|████████| 15/15 [00:04<00:00]

Index ready — all products embedded
Product Tracker Agent is live

Which products are performing best?

## TOP PERFORMERS

• Smart Fitness Watch — rating: 4.7, last_month_sales: 312
  → Meets TOP PERFORMER criteria (rating ≥ 4.6, sales > 200)

• Wireless Noise Cancelling Headphones — rating: 4.6, last_month_sales: 245
  → Meets TOP PERFORMER criteria

...

## PRIORITY ACTIONS
1. ...
2. ...
3. ...
```

---

## Customization

### Change the query

At the bottom of `Reporter.py`, edit the `.query()` call:

```python
response = tracker.query(
    'Which products are at risk of going out of stock this week?'
)
```

Other useful queries to try:
- `"List all products that need reordering and their suppliers."`
- `"Which products are a supply risk? Explain the urgency for each."`
- `"Give me a full inventory health report."`

### Swap the LLM

Replace `gemma4:e2b` with any model you have pulled in Ollama:

```python
Settings.llm = Ollama(
    model='llama3.2',   # or mistral, phi3, etc.
    request_timeout=300.0,
    timeout=300.0,
    keep_alive='60m',
)
```

### Adjust retrieval breadth

`similarity_top_k=15` retrieves all 15 products. Lower this for larger catalogs if you only want the most relevant results per query:

```python
tracker = index.as_query_engine(
    text_qa_template=PRODUCT_TRACKER_PROMPT,
    similarity_top_k=5,
)
```

---

## Troubleshooting

### `httpx.ReadTimeout` error

The LLM took too long to respond. Increase the timeout or ensure the warm-up call runs before the query:

```python
Settings.llm = Ollama(
    model='gemma4:e2b',
    request_timeout=300.0,   # increase if still timing out
    timeout=300.0,
    keep_alive='60m',
)
Settings.llm.complete('hi')  # warm-up — must come before building the index
```

### `ConnectionRefusedError` or Ollama not responding

Ollama is not running. Start it with:
```bash
ollama serve
```

### Model not found

You haven't pulled the model yet:
```bash
ollama pull gemma4:e2b
ollama pull nomic-embed-text
```

### CSV file not found

Make sure `products.csv` is in the **same directory** as `Reporter.py` when you run the script, or pass the full path:
```python
nodes = build_nodes_from_csv('C:/full/path/to/products.csv')
```

---

## Tech Stack

| Component | Tool |
|---|---|
| RAG Framework | [LlamaIndex](https://www.llamaindex.ai/) |
| LLM | [Ollama](https://ollama.com/) + `gemma4:e2b` |
| Embeddings | [Ollama](https://ollama.com/) + `nomic-embed-text` |
| Vector Store | LlamaIndex in-memory `VectorStoreIndex` |
| Data Source | CSV via Python `csv.DictReader` |

---

## License

MIT License — free to use, modify, and distribute.
