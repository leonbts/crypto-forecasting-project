## Quick start (local inference)

```bash
docker build -t crypto-forecast:inference ./inference_image
docker run --rm -v /path/to/model:/model -v /path/to/input.csv:/input.csv crypto-forecast:inference
```