## Quick start (local inference)

```bash
docker build -t crypto-forecast:inference ./inference_image
docker run --rm -v /path/to/model:/model -v /path/to/input.csv:/input.csv crypto-forecast:inference
```


# 📌 Project Milestones & Roadmap

## ✅ Completed Milestones

### 1. End-to-end data ingestion pipeline
- Automated collection of daily cryptocurrency OHLCV data from an external API.
- Raw data stored in Amazon S3 using a date-partitioned structure.
- Ingestion logic designed to be idempotent, preventing data corruption or duplication.

---

### 2. Data preprocessing and feature preparation
- Daily preprocessing Lambda converts raw JSON files into clean, deduplicated CSV datasets.
- Standardized schema across all assets (open, high, low, close, volume).
- Processed datasets stored in S3 for downstream consumption.

---

### 3. Model training (offline)
- Supervised regression model trained on historical daily OHLCV data.
- Sliding-window formulation used to capture short-term temporal dependencies.
- Time-based train/validation split to avoid data leakage.
- Trained model artifacts (weights, scalers, metadata) persisted to S3.
- Training executed in a managed AWS environment (Amazon SageMaker).

---

### 4. Containerized inference pipeline
- Inference logic packaged into a Docker image for reproducibility.
- Image stored in Amazon ECR with versioned tags.
- Serverless inference execution triggered via AWS Lambda.
- Predictions written to S3 for later analysis and evaluation.

---

## ⏳ Planned / Future Work

### 5. Model monitoring and evaluation (Production ML)
- Introduce a daily monitoring Lambda that evaluates model predictions against actual observed prices.
- Compute and track the following metrics:
  - Mean Absolute Error (MAE)
  - Directional Accuracy (DA)
- Compare model performance against naïve baselines (e.g., persistence model).
- Store per-day and rolling evaluation metrics (7-day and 30-day windows) in S3.
- Use monitoring results to detect performance degradation, data issues, and market regime shifts.

---

### 6. Automated retraining pipeline (SageMaker)
- Implement periodic or performance-triggered retraining using newly available data.
- Retraining executed as a SageMaker Training Job.
- Versioned model artifacts stored in S3 with explicit promotion to a models/latest/ directory.
- Existing inference pipeline remains unchanged, ensuring clear separation between training and serving.

---

## 🎯 Design Philosophy

- Separation of concerns: ingestion, preprocessing, inference, and (future) retraining are decoupled.
- Reproducibility: Docker and SageMaker ensure consistent execution across environments.
- Scalability: serverless components minimize operational overhead.
- Observability: monitoring provides transparency into real-world model performance.