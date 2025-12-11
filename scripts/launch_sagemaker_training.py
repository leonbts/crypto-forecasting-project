import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from pathlib import Path

# ====== CONFIG - EDIT THESE ======

# Your SageMaker execution role ARN (from IAM console)
ROLE_ARN = "arn:aws:iam::546602404979:role/sagemaker-crypto-training-role"

# Your S3 bucket (where processed data + models live)
BUCKET_NAME = "crypto-forecast-bucket"

# S3 locations
TRAIN_DATA_S3 = f"s3://{BUCKET_NAME}/processed/daily/"
OUTPUT_S3 = f"s3://{BUCKET_NAME}/models/sagemaker/"

# Instance type for training
INSTANCE_TYPE = "ml.m5.large"  # or "ml.m5.xlarge" if you want more power

# SageMaker region (must match your bucket region)
REGION = "eu-central-1"

# Framework settings
TF_FRAMEWORK_VERSION = "2.12"
PYTHON_VERSION = "py39"

# Job name prefix
BASE_JOB_NAME = "crypto-lstm-training"


def main():
    # Resolve script paths
    project_root = Path(__file__).resolve().parents[1]
    source_dir = project_root / "src" / "sagemaker"
    entry_point = "train_lstm_sagemaker.py"

    print(f"[INFO] Using source_dir: {source_dir}")
    print(f"[INFO] Entry point: {entry_point}")
    print(f"[INFO] Training data S3: {TRAIN_DATA_S3}")
    print(f"[INFO] Output S3: {OUTPUT_S3}")

    # SageMaker session & client
    session = sagemaker.Session(boto3.Session(region_name=REGION))

    # Define hyperparameters (these are passed to argparse in train_lstm_sagemaker.py)
    hyperparameters = {
        "window-size": 60,
        "horizon": 7,
        "epochs": 20,
        "batch-size": 64,
        "train-fraction": 0.7,
        "val-fraction": 0.15,
        "learning-rate": 1e-3,
        "random-seed": 42,
    }

    estimator = TensorFlow(
        entry_point=entry_point,
        source_dir=str(source_dir),
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version=TF_FRAMEWORK_VERSION,
        py_version=PYTHON_VERSION,
        hyperparameters=hyperparameters,
        output_path=OUTPUT_S3,
        base_job_name=BASE_JOB_NAME,
        # Optional: use script mode, no training script in image itself
        # script_mode is default in new SDKs, so we don't need to set it.
    )

    # Launch training
    print("[INFO] Starting SageMaker training job...")
    estimator.fit({"train": TRAIN_DATA_S3})
    print("[INFO] Training job submitted. Check the SageMaker console for status.")


if __name__ == "__main__":
    main()