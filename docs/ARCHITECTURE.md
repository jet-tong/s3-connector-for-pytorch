# Architecture Overview

The Amazon S3 Connector for PyTorch is **not a new S3 client**. It's a purpose-built bridge between
PyTorch's data loading and checkpointing interfaces and the high-performance
[mountpoint-s3-client](https://github.com/awslabs/mountpoint-s3) Rust library (built on the AWS Common Runtime).

It exists to deliver streaming, high-throughput S3 access (10+ Gbps) directly into PyTorch workflows —
without local disk staging, manual multipart handling, or GIL contention.

---

## Architecture Stack

![Architecture Stack](architecture-stack.svg)

| Layer | Role |
|-------|------|
| **s3torchconnector** (Python) | PyTorch-native interfaces — datasets, checkpoints, reader strategies, fork safety |
| **s3torchconnectorclient** (PyO3) | Thin bridge exposing the Rust S3 client as Python classes |
| **mountpoint-s3-client** (Rust) | Streaming S3 GET/PUT with automatic multipart, retries, connection pooling |
| **AWS CRT** (C) | Low-level HTTP, TLS, DNS, and S3 transfer primitives — auto-tunes parallelism |
| **Amazon S3** | Object storage service |

---

## How Rust Connects to Pythonv

The Rust layer is exposed to Python via [PyO3](https://pyo3.rs/) bindings. The key class is
`MountpointS3Client`, which provides `get_object`, `put_object`, `list_objects`, and other S3 operations.

**GIL release during I/O** — Every S3 call releases the Python Global Interpreter Lock, so Python
threads (including DataLoader workers) aren't blocked while waiting on network I/O.

**Streaming interfaces:**
- `GetObjectStream` — a Python iterator yielding ~8MB byte chunks from S3
- `PutObjectStream` — accepts `write()` calls, streaming data to S3 via multipart upload

**Fork safety for DataLoader workers** — PyTorch's DataLoader forks worker processes. The connector
detects PID changes on every S3 access and lazily re-creates the native client in the child process.
CRT background threads are cleaned up before fork via `os.register_at_fork`. This is automatic —
no user action required.

---

## Interface Mapping

```mermaid
classDiagram
    %% Data Loading
    class Dataset {
        <<torch.utils.data>>
        +__getitem__(index)
        +__len__()
    }
    class IterableDataset {
        <<torch.utils.data>>
        +__iter__()
    }
    class S3MapDataset {
        +from_prefix(uri, region)$
        +__getitem__(i) S3Reader
        +__len__() int
    }
    class S3IterableDataset {
        +from_prefix(uri, region)$
        +__iter__() Iterator~S3Reader~
    }
    Dataset <|-- S3MapDataset
    IterableDataset <|-- S3IterableDataset

    %% I/O Streams
    class BufferedIOBase {
        <<io>>
        +read()
        +write()
        +seek()
    }
    class S3Reader {
        +read() bytes
        +seek(offset)
        +readinto(buf)
    }
    class S3Writer {
        +write(data)
        +close()
    }
    BufferedIOBase <|-- S3Reader
    BufferedIOBase <|-- S3Writer

    %% DCP (Distributed Checkpointing)
    class FileSystemBase {
        <<torch.distributed.checkpoint>>
        +create_stream(path, mode)
        +rename(old, new)
        +exists(path)
        +rm_file(path)
    }
    class FileSystemWriter {
        <<torch.distributed.checkpoint>>
    }
    class FileSystemReader {
        <<torch.distributed.checkpoint>>
    }
    class S3FileSystem {
        +create_stream(path, mode)
        +concat_path(path, suffix)
        +init_path(path)
        +rename(old, new)
        +exists(path) bool
        +rm_file(path)
        +validate_checkpoint_id(id)$ bool
    }
    class S3StorageWriter {
        +prepare_global_plan(plans)
        +validate_checkpoint_id(id)$ bool
    }
    class S3StorageReader {
        +prepare_local_plan(plan)
        +validate_checkpoint_id(id)$ bool
    }
    FileSystemBase <|-- S3FileSystem
    FileSystemWriter <|-- S3StorageWriter
    FileSystemReader <|-- S3StorageReader
    S3StorageWriter *-- S3FileSystem : fs
    S3StorageReader *-- S3FileSystem : fs

    %% Lightning
    class CheckpointIO {
        <<lightning>>
        +save_checkpoint()
        +load_checkpoint()
        +remove_checkpoint()
    }
    class S3LightningCheckpoint {
        +save_checkpoint(checkpoint, path)
        +load_checkpoint(path)
        +remove_checkpoint(path)
    }
    CheckpointIO <|-- S3LightningCheckpoint
```

| S3 Connector Class | Extends | Purpose |
|---|---|---|
| `S3MapDataset` | `torch.utils.data.Dataset` | Random access — eagerly lists objects, supports `__getitem__` |
| `S3IterableDataset` | `torch.utils.data.IterableDataset` | Streaming — lazy iteration with automatic sharding |
| `S3Reader` / `S3Writer` | `io.BufferedIOBase` | Drop-in compatible with `torch.save()` / `torch.load()` |
| `S3FileSystem` | `FileSystemBase` | S3 filesystem operations for DCP |
| `S3StorageWriter` | `FileSystemWriter` | Distributed checkpoint save to S3 |
| `S3StorageReader` | `FileSystemReader` | Distributed checkpoint load from S3 |
| `S3LightningCheckpoint` | `CheckpointIO` | PyTorch Lightning checkpoint plugin |

---

## Reader Hierarchy

The connector provides three reader strategies, selectable via `S3ReaderConstructor`:

```mermaid
classDiagram
    class S3Reader {
        <<abstract>>
        +bucket: str
        +key: str
        +read(size) bytes
        +readinto(buf) int
        +seek(offset, whence) int
    }

    class SequentialS3Reader {
        Buffers entire object
        Best for full-object reads
    }

    class RangedS3Reader {
        Byte-range requests
        Best for sparse partial reads
    }

    class DCPOptimizedS3Reader {
        Range coalescing + zero-copy
        Best for DCP checkpoint loading
    }

    class S3ReaderConstructor {
        +sequential()$
        +range_based(buffer_size)$
        +dcp_optimized()$
    }

    S3Reader <|-- SequentialS3Reader
    S3Reader <|-- RangedS3Reader
    S3Reader <|-- DCPOptimizedS3Reader
    S3ReaderConstructor ..> SequentialS3Reader : creates
    S3ReaderConstructor ..> RangedS3Reader : creates
    S3ReaderConstructor ..> DCPOptimizedS3Reader : creates
```

| Reader | Strategy | Default for |
|--------|----------|-------------|
| **SequentialS3Reader** | Downloads and buffers the entire object; fast random access after first read | Datasets, simple checkpoints |
| **RangedS3Reader** | Byte-range requests; only fetches what's needed; adaptive 8MB buffer | Explicit opt-in for large sparse reads |
| **DCPOptimizedS3Reader** | Groups nearby byte ranges into single S3 requests; zero-copy via memoryview | `S3StorageReader` (automatic) |

For detailed reader configuration and usage examples, see the
[Reader Configurations](READER_CONFIGURATIONS.md) documentation.

---

## Key Design Decisions

### Why mountpoint-s3-client (not boto3)

boto3's transfer manager doesn't support true streaming — it downloads to disk or memory first.
mountpoint-s3-client provides native streaming GET/PUT built on the AWS CRT, delivering 10+ Gbps
throughput with automatic multipart, connection pooling, and retry logic.

### Why PyO3 (not ctypes/cffi)

PyO3 provides compile-time type safety, explicit GIL control (critical for releasing the GIL during
I/O), and native async support. ctypes/cffi would require manual memory management and offer no
GIL release mechanism.

### Why lazy client initialization

PyTorch's DataLoader forks worker processes. If the S3 client were initialized eagerly, CRT
background threads would be duplicated across forks — causing crashes or hangs. Lazy init (triggered
on first use) ensures each process creates its own client after fork.

### Why the reader_constructor pattern

Decouples the read strategy from the PyTorch interface. The same `S3MapDataset`, `S3Checkpoint`, or
`S3StorageReader` can use any reader strategy without code changes — just pass a different
`S3ReaderConstructor`. This also allows the DCP path to inject range metadata without modifying the
reader interface.

---

## Further Reading

- [README](../README.md) — Usage examples and getting started
- [Troubleshooting](TROUBLESHOOTING.md) — Common issues and solutions
- [API Documentation](https://awslabs.github.io/s3-connector-for-pytorch) — Full API reference
- [dev-docs/onboarding/](../dev-docs/onboarding/) — Deep internals for contributors
