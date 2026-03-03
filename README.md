# SME Transaction Categoriser

An AI-powered transaction categorisation service specifically designed for UK SME bank transactions. It employs a rigorous, tiered machine learning approach to accurately classify banking transactions into predefined financial categories.

## Architecture & Methodology

The system relies on a well-optimised 3-tier pipeline. This progressive approach ensures high precision for known patterns whilst retaining the flexibility to categorise complex or unseen transaction descriptions.

1. **Tier 1 (Exact & Fuzzy Matching)** 
   The fastest categorisation tier. It relies on a pre-computed exact lookup table built during the training phase. If an exact match is not found, the system employs `rapidfuzz` string similarity to handle minor typos or slight variations. This tier operates within strict directional guardrails (DEBIT/CREDIT/BOTH).

2. **Tier 2 (Keyword-Filtered Semantic Search)** 
   For transactions that bypass Tier 1, the system falls back to a semantic approach. Using `sentence-transformers`, descriptions are converted into embeddings and queried against a Qdrant vector database. 
   - **Direct Keyword Hits:** If extremely high-confidence keywords are found (e.g., scoring > 92%), the system categorises the transaction immediately.
   - **Semantic Validation:** For medium-confidence keywords, Qdrant semantic search validates the category. It factors in vector similarity scores and score gaps to guarantee maximum reliability.

3. **Tier 3 (XGBoost Fallback)** 
   The final tier operates as a safety net for complex, ambiguous cases. It utilises an XGBoost machine learning model trained on transaction signals, textual embeddings, word/character counts, and transaction amounts (e.g., highlighting anomalous large debits or credits). Like the other tiers, this model strictly masks out categories that violate the DEBIT/CREDIT direction guardrails.

## Project Structure

A brief overview of the primary directories and their responsibilities:

- `config/`: Configuration files mapping core rules, such as `config.yaml` and `feature_signals.json`.
- `data/`: The central hub for all dataset handling. Contains raw/synthetic CSV data, category trees, compiled model artefacts, and the persistent Qdrant index.
- `scripts/`: Operational CLI scripts facilitating model training, synthetic data generation, and robust performance evaluation.
- `src/categoriser/`: The core application repository.
  - `api/`: A FastAPI web server that provides endpoints for both single and grouped batch categorisation requests.
  - `engine/`: Contains the central `orchestrator.py` which unifies the 3-tier logic into a seamless pipeline.

## Usage Guide

Ensure your Python environment is active and the dependencies (such as `xgboost`, `qdrant-client`, `sentence-transformers`, and `fastapi`) are installed before executing these commands.

### 1. Training the Model

Before making inferences, you must train the model using your transaction dataset. This aggregates your exact matching lookups, populates the Qdrant indexing database, and trains the XGBoost model.

```bash
python scripts/train_models.py --version v1.0
```

### 2. Evaluating the Model

Once trained, it is highly recommended to run the evaluation script against the test data split. This provides an exhaustive breakdown of accuracy, F1 scores per categorisation group, and an overarching confusion matrix.

```bash
python scripts/evaluate_models.py --version v1.0
```
*Outputs for the evaluation will be saved in the respective `data/versions/{version}/evaluation/` directory.*

### 3. Running the REST API

You can start the FastAPI application to serve the model capabilities over an HTTP server.

```bash
python -m uvicorn categoriser.api.main:app --reload --host 0.0.0.0 --port 8000
```

Navigate to `http://localhost:8000/docs` in your web browser to view the interactive Swagger API documentation.

### 4. Running the CLI Orchestrator (Batch Processing)

For offline categorisation, the engine exposes a command-line interface capable of batch processing entire CSV files of transactions.

```bash
python src/categoriser/engine/orchestrator.py --version v1.0 --input data/test.csv --output categorised_results.csv
```
