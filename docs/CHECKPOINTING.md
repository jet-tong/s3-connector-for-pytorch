# Checkpointing

Save and load PyTorch model checkpoints directly to Amazon S3, without first saving to local storage.

## Table of Contents

- [Basic Usage](#basic-usage)
- [S3 Express One Zone](#s3-express-one-zone)
- [Lightning Integration](#lightning-integration)
- [S3 Versioning for Checkpoints](#s3-versioning-for-checkpoints)
- [Distributed Checkpoints](#distributed-checkpoints)

## Basic Usage

The `S3Checkpoint` class provides `reader()` and `writer()` context managers that return streams compatible with `torch.save` and `torch.load`.

### Construction

```py
from s3torchconnector import S3Checkpoint

checkpoint = S3Checkpoint(region="us-east-1")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `region` | `str` | AWS region of the S3 bucket |
| `endpoint` | `str`, optional | Custom S3 endpoint URL |
| `s3client_config` | `S3ClientConfig`, optional | Client configuration (see [CONFIGURATION.md](CONFIGURATION.md)) |

### Saving a Checkpoint

Use `checkpoint.writer(s3_uri)` to get a write-only binary stream. Pass it directly to `torch.save`:

```py
with checkpoint.writer("s3://my-bucket/checkpoints/epoch0.ckpt") as writer:
    torch.save(model.state_dict(), writer)
```

### Loading a Checkpoint

Use `checkpoint.reader(s3_uri)` to get a read-only binary stream. Pass it directly to `torch.load`:

```py
with checkpoint.reader("s3://my-bucket/checkpoints/epoch0.ckpt") as reader:
    state_dict = torch.load(reader)
```

### Full Example

```py
from s3torchconnector import S3Checkpoint

import torchvision
import torch

CHECKPOINT_URI = "s3://my-bucket/checkpoints/"
REGION = "us-east-1"
checkpoint = S3Checkpoint(region=REGION)

model = torchvision.models.resnet18()

# Save checkpoint to S3
with checkpoint.writer(CHECKPOINT_URI + "epoch0.ckpt") as writer:
    torch.save(model.state_dict(), writer)

# Load checkpoint from S3
with checkpoint.reader(CHECKPOINT_URI + "epoch0.ckpt") as reader:
    state_dict = torch.load(reader)

model.load_state_dict(state_dict)
```

## S3 Express One Zone

To use checkpoints with [Amazon S3 Express One Zone](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-express-one-zone.html) directory buckets, update the URI to use the `base-name--azid--x-s3` bucket name format.

For example, with a directory bucket named `my-test-bucket--usw2-az1--x-s3` in Availability Zone ID `usw2-az1`:

```py
checkpoint = S3Checkpoint(region="us-west-2")

# Note: prefix for S3 Express One Zone should end with '/'
CHECKPOINT_URI = "s3://my-test-bucket--usw2-az1--x-s3/checkpoints/"

with checkpoint.writer(CHECKPOINT_URI + "epoch0.ckpt") as writer:
    torch.save(model.state_dict(), writer)
```

> **Note:** S3 Versioning and S3 Lifecycle are not supported by S3 Express One Zone.

## Lightning Integration

Amazon S3 Connector for PyTorch includes `S3LightningCheckpoint`, an implementation of Lightning's `CheckpointIO` that enables saving and loading checkpoints directly to S3.

### Installation

```sh
pip install s3torchconnector[lightning]
```

### Usage

```py
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from s3torchconnector.lightning import S3LightningCheckpoint

# Create the S3 checkpoint plugin
s3_checkpoint_io = S3LightningCheckpoint("us-east-1")

# Configure checkpoint saving behavior
checkpoint_callback = ModelCheckpoint(
    dirpath="s3://my-bucket/checkpoints/",
    save_top_k=1,
    every_n_train_steps=1,
    filename="checkpoint-{epoch:02d}-{step:02d}",
)

# Pass the plugin to the Trainer
trainer = Trainer(
    plugins=[s3_checkpoint_io],
    callbacks=[checkpoint_callback],
    max_epochs=5,
)
trainer.fit(model, dataloader)
```

For complete examples, see the [examples/lightning](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/examples/lightning) directory.

## S3 Versioning for Checkpoints

When working with model checkpoints, you can use the [S3 Versioning](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html) feature to preserve, retrieve, and restore every version of your checkpoint objects. With versioning, you can recover more easily from unintended overwrites or deletions of existing checkpoint files due to incorrect configuration or multiple hosts accessing the same storage path.

When versioning is enabled on an S3 bucket, deletions insert a delete marker instead of removing the object permanently. The delete marker becomes the current object version. If you overwrite an object, it results in a new object version in the bucket. You can always restore the previous version. See [Deleting object versions from a versioning-enabled bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/DeletingObjectVersions.html) for more details on managing object versions.

To enable versioning on an S3 bucket, see [Enabling versioning on buckets](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html). Normal Amazon S3 rates apply for every version of an object stored and transferred. To customize your data retention approach and control storage costs for earlier versions of objects, use [object versioning with S3 Lifecycle](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html).

> **Note:** S3 Versioning and S3 Lifecycle are not supported by S3 Express One Zone.

## Distributed Checkpoints

For distributed checkpoint support using PyTorch's `torch.distributed.checkpoint` (DCP), including `S3StorageWriter`, `S3StorageReader`, and prefix strategies for high-throughput scenarios, see [DISTRIBUTED_CHECKPOINTS.md](DISTRIBUTED_CHECKPOINTS.md).
