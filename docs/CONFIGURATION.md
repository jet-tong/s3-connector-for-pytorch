# Configuration Reference

This document covers all configuration options for Amazon S3 Connector for PyTorch, including AWS credentials, client tuning, and advanced usage.

## Table of Contents

- [AWS Credentials Configuration](#aws-credentials-configuration)
- [S3ClientConfig Reference](#s3clientconfig-reference)
- [S3 Express One Zone](#s3-express-one-zone)
- [Direct S3Client Usage](#direct-s3client-usage)
- [Performance Tuning Guide](#performance-tuning-guide)

## AWS Credentials Configuration

AWS credentials must be provided through one of the following methods:

### EC2 Instance Role

If running on an EC2 instance, attach an IAM role with the necessary S3 permissions to the instance. No additional configuration is required — credentials are automatically retrieved from the instance metadata service.

### AWS CLI

Install and configure the [AWS CLI](https://aws.amazon.com/cli/):

```shell
aws configure
```

This stores credentials in `~/.aws/credentials` and configuration in `~/.aws/config`.

### Credential Files

Set credentials directly in `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

### Environment Variables

```shell
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
```

### Custom Profile

To use a named profile configured in `~/.aws/config` and `~/.aws/credentials`:

**Option 1** — Environment variable:

```shell
export AWS_PROFILE=custom-profile
```

**Option 2** — Pass directly to `S3ClientConfig`:

```python
from s3torchconnector import S3ClientConfig

config = S3ClientConfig(profile="custom-profile")
dataset = S3MapDataset.from_prefix(uri, region=region, s3client_config=config)
```

For detailed credential configuration, see the [AWS CLI configuration docs](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html).

## S3ClientConfig Reference

`S3ClientConfig` exposes tunable parameters for the underlying S3 CRT client.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `throughput_target_gbps` | `float` | `10.0` | Throughput target in Gigabits per second (Gbps) |
| `part_size` | `int` | `8388608` (8 MiB) | Size in bytes of multipart upload/download chunks. Must be between 5 MiB and 5 GiB |
| `unsigned` | `bool` | `False` | Disable request signing (for public buckets) |
| `force_path_style` | `bool` | `False` | Use path-style S3 addressing instead of virtual-hosted |
| `max_attempts` | `int` | `10` | Number of retry attempts for retriable errors |
| `profile` | `Optional[str]` | `None` | AWS credential profile name |

### Usage

```python
from s3torchconnector import S3MapDataset, S3Checkpoint, S3ClientConfig

DATASET_URI = "s3://my-bucket/training-data/"
REGION = "us-east-1"

# Custom configuration
config = S3ClientConfig(
    throughput_target_gbps=100,
    part_size=16 * 1024 * 1024,
    max_attempts=5,
)

# Pass to datasets
dataset = S3MapDataset.from_prefix(DATASET_URI, region=REGION, s3client_config=config)

# Pass to checkpoints
checkpoint = S3Checkpoint(region=REGION, s3client_config=config)
```

### Unsigned Requests

For public buckets that don't require authentication:

```python
config = S3ClientConfig(unsigned=True)
dataset = S3MapDataset.from_prefix(DATASET_URI, region=REGION, s3client_config=config)
```

> **Note:** When modifying defaults, run benchmarking to ensure you are not introducing a performance regression.

## S3 Express One Zone

[Amazon S3 Express One Zone](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-express-one-zone.html) directory buckets are supported across all interfaces (datasets, checkpoints, and distributed checkpoints).

### URI Format

Directory bucket names follow the format `base-name--azid--x-s3`. The prefix **must** end with a trailing `/`:

```
s3://my-bucket--usw2-az1--x-s3/prefix/
```

### Example

```python
from s3torchconnector import S3MapDataset

# Directory bucket in us-west-2, Availability Zone usw2-az1
DATASET_URI = "s3://my-bucket--usw2-az1--x-s3/training-data/"
REGION = "us-west-2"

dataset = S3MapDataset.from_prefix(DATASET_URI, region=REGION)
```

> **Important:** The prefix for S3 Express One Zone must end with `/`.

## Direct S3Client Usage

For advanced use cases requiring custom streaming patterns, use `S3Client` directly:

```python
from s3torchconnector._s3client import S3Client

REGION = "us-east-1"
BUCKET_NAME = "my-bucket"
OBJECT_KEY = "large_object.bin"

s3_client = S3Client(region=REGION)

# Writing data to S3
data = b"content" * 1048576
s3writer = s3_client.put_object(bucket=BUCKET_NAME, key=OBJECT_KEY)
s3writer.write(data)
s3writer.close()

# Reading data from S3
s3reader = s3_client.get_object(bucket=BUCKET_NAME, key=OBJECT_KEY)
data = s3reader.read()
```

`S3Client` also accepts `S3ClientConfig` for tuning:

```python
from s3torchconnector._s3client import S3Client
from s3torchconnector import S3ClientConfig

config = S3ClientConfig(throughput_target_gbps=50)
s3_client = S3Client(region=REGION, s3client_config=config)
```

## Performance Tuning Guide

### Throughput Target (`throughput_target_gbps`)

The default of 10 Gbps is suitable for most instance types. Increase this value when running on larger instances with higher network bandwidth:

| Instance Network | Recommended Target |
|------------------|--------------------|
| Up to 10 Gbps | `10.0` (default) |
| 25 Gbps | `25.0` |
| 100 Gbps (e.g., p4d, p5) | `100.0` |

```python
# For p4d/p5 instances with 100 Gbps networking
config = S3ClientConfig(throughput_target_gbps=100)
```

### Part Size (`part_size`)

The default 8 MiB part size works well for most workloads. Consider adjusting when:

- **Larger parts** (16–64 MiB): Reduces per-request overhead for large sequential transfers. Useful for checkpoint writes of large models.
- **Smaller parts** (5 MiB minimum): May improve time-to-first-byte for smaller objects.

```python
# Larger parts for big checkpoint writes
config = S3ClientConfig(part_size=64 * 1024 * 1024)
```

> **Note:** For checkpoint saves, the client automatically adjusts part size to meet S3 service limits (max 10,000 parts per upload, minimum 5 MiB per part).

### Reader Selection

Different reader types optimize for different access patterns. See [Reader Configurations](READER_CONFIGURATIONS.md) for details on choosing between sequential, range-based, and DCP-optimized readers.

### DCP Write Parallelism

For distributed checkpoint writes, the `thread_count` parameter on `S3StorageWriter` controls I/O parallelism. See [Distributed Checkpoints](DISTRIBUTED_CHECKPOINTS.md) for configuration guidance.
