# Amazon S3 Connector for PyTorch

The Amazon S3 Connector for PyTorch delivers high throughput for PyTorch training jobs that access or store data in Amazon S3. Built on the AWS Common Runtime (CRT), it eliminates the need to write custom code for S3 listing, concurrent transfers, and checkpoint staging — automatically optimizing performance for 10+ Gbps throughput.

## Features

- High-throughput data loading from S3 (map-style and iterable datasets)
- Direct checkpoint save/load to S3 (no local disk staging)
- PyTorch Distributed Checkpoint (DCP) support with S3 prefix strategies
- PyTorch Lightning integration
- Configurable reader strategies (sequential, range-based, DCP-optimized)
- Built on AWS CRT for 10+ Gbps throughput

## Getting Started

### Prerequisites

- Python 3.8–3.14 (Note: Python 3.8 support will be deprecated in a future release, see [#399](https://github.com/awslabs/s3-connector-for-pytorch/issues/399))
- PyTorch >= 2.0

### Installation

```shell
pip install s3torchconnector
```

Pre-built wheels are available for Linux and macOS (ARM64). (Note: macOS x86_64 wheel support will be deprecated in a future release, see [#398](https://github.com/awslabs/s3-connector-for-pytorch/issues/398))

### Quick Example

**Load training data from S3:**

```python
from s3torchconnector import S3IterableDataset

dataset = S3IterableDataset.from_prefix("s3://BUCKET/PREFIX", region="us-east-1")
for item in dataset:
    content = item.read()
```

**Save/load checkpoints directly to S3:**

```python
from s3torchconnector import S3Checkpoint
import torch

checkpoint = S3Checkpoint(region="us-east-1")

with checkpoint.writer("s3://BUCKET/model.ckpt") as writer:
    torch.save(model.state_dict(), writer)

with checkpoint.reader("s3://BUCKET/model.ckpt") as reader:
    state_dict = torch.load(reader)
```

## Documentation

- [Configuration](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/docs/CONFIGURATION.md) — AWS credentials, profiles, S3 Express One Zone, and client options
- [Datasets](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/docs/DATASETS.md) — Map-style and iterable datasets, parallel/distributed training
- [Checkpointing](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/docs/CHECKPOINTING.md) — Single-device checkpoint save/load, S3 Versioning
- [Distributed Checkpoints](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/docs/DISTRIBUTED_CHECKPOINTS.md) — DCP integration, prefix strategies, S3FileSystem
- [Reader Configurations](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/docs/READER_CONFIGURATIONS.md) — Sequential, range-based, and DCP-optimized readers
- [Lightning Integration](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/docs/CHECKPOINTING.md#lightning-integration) — PyTorch Lightning CheckpointIO plugin
- [Benchmarking](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/s3torchbenchmarking/README.md) — Performance testing and tuning
- [Troubleshooting](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/docs/TROUBLESHOOTING.md) — Common issues and solutions

### API Reference

Full API documentation: [awslabs.github.io/s3-connector-for-pytorch](https://awslabs.github.io/s3-connector-for-pytorch)

### Examples

End-to-end examples: [`examples/`](https://github.com/awslabs/s3-connector-for-pytorch/tree/main/examples)

## Contributing

We welcome contributions to Amazon S3 Connector for PyTorch. Please see [CONTRIBUTING.md](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/CONTRIBUTING.md) for more information on how to report bugs or submit pull requests.

### Development

See [DEVELOPMENT.md](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/DEVELOPMENT.md) for information about code style, development process, and guidelines.

### Compatibility with other storage services

S3 Connector for PyTorch delivers high throughput for PyTorch training jobs that access or store data in Amazon S3. While it may be functional against other storage services that use S3-like APIs, they may inadvertently break when we make changes to better support Amazon S3. We welcome contributions of minor compatibility fixes or performance improvements for these services if the changes can be tested against Amazon S3.

### Security issue notifications

If you discover a potential security issue in this project we ask that you notify AWS Security via our [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/).

### Code of conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). See [CODE_OF_CONDUCT.md](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/CODE_OF_CONDUCT.md) for more details.

## License

Amazon S3 Connector for PyTorch has a BSD 3-Clause License, as found in the [LICENSE](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/LICENSE) file.
