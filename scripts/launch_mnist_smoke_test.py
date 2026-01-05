import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from pathlib import Path

ROLE_ARN = "arn:aws:iam::546602404979:role/sagemaker-crypto-training-role"
BUCKET_NAME = "crypto-forecast-bucket"
REGION = "eu-central-1"

INSTANCE_TYPE = "ml.m4.xlarge"   # use the one you have quota for
TF_FRAMEWORK_VERSION = "2.12"
PYTHON_VERSION = "py310"

BASE_JOB_NAME = "smoke-test-mnist-tf"


def main():
    project_root = Path(__file__).resolve().parents[1]
    source_dir = project_root / "src" / "sagemaker"

    boto_sess = boto3.Session(region_name=REGION)
    session = sagemaker.Session(boto_session=boto_sess)

    estimator = TensorFlow(
        entry_point="mnist_tf_train.py",
        source_dir=str(source_dir),
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version=TF_FRAMEWORK_VERSION,
        py_version=PYTHON_VERSION,
        output_path=f"s3://{BUCKET_NAME}/models/sagemaker/",
        base_job_name=BASE_JOB_NAME,
        sagemaker_session=session,
    )

    # No channels needed; MNIST loads inside the container
    estimator.fit()
    print("[INFO] Smoke test submitted.")


if __name__ == "__main__":
    main()