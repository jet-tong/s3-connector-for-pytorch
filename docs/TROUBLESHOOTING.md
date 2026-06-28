# Troubleshooting

## Getting Help

If `s3torchconnector` is not working as expected, please check [GitHub Issues](https://github.com/awslabs/s3-connector-for-pytorch/issues) to see if your issue has already been addressed.

If not, [create a new issue](https://github.com/awslabs/s3-connector-for-pytorch/issues/new/choose) with the following details:

- **Version**: `pip show s3torchconnector` output
- **Python/PyTorch versions**: `python --version` and `torch.__version__`
- **Environment**: OS, instance type (if EC2), number of GPUs/nodes
- **Minimal reproduction**: Smallest code snippet that triggers the issue
- **Error output**: Full traceback and any relevant logs (see [Debug Logging](#debug-logging))

---

## Debug Logging

The S3 Connector includes Rust-level logging for the underlying S3 operations (mountpoint-s3-client and AWS CRT). These logs are disabled by default.

### Enabling Logs

| Environment Variable | Purpose |
|---------------------|---------|
| `S3_TORCH_CONNECTOR_DEBUG_LOGS` | Log level filter (same syntax as [RUST_LOG](https://docs.rs/env_logger/latest/env_logger/#enabling-logging)) |
| `S3_TORCH_CONNECTOR_LOGS_DIR_PATH` | Directory for log file output. If unset, logs go to stderr. |

### Examples

```bash
# INFO-level logs to stderr
export S3_TORCH_CONNECTOR_DEBUG_LOGS=info

# TRACE-level logs to file (most verbose, excludes noisy CRT logs)
export S3_TORCH_CONNECTOR_DEBUG_LOGS="trace,awscrt=off"
export S3_TORCH_CONNECTOR_LOGS_DIR_PATH="/tmp/s3torchconnector-logs"

# Fine-grained component control
export S3_TORCH_CONNECTOR_DEBUG_LOGS="trace,mountpoint_s3_client=debug,awscrt=error"
```

### What the Logs Contain

- S3 request/response details (GET, PUT, LIST, HEAD operations)
- Multipart upload progress
- Connection lifecycle events
- Retry attempts and errors
- CRT-level networking details (when not filtered with `awscrt=off`)

Log files are written to `${S3_TORCH_CONNECTOR_LOGS_DIR_PATH}/s3torchconnectorclient.log.yyyy-MM-dd-HH` and rolled hourly.

> **Note**: AWS CRT logs are very noisy. We recommend filtering them with `awscrt=off` unless debugging low-level networking issues.

For additional debugging tools (GDB, Python debuggers), see the [Debugging section in DEVELOPMENT.md](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/DEVELOPMENT.md#debugging).

---

## Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| "No signing credentials found" / `CredentialsProviderError` | Missing or expired AWS credentials | Check AWS config (see [Credentials Issues](#credentials-issues)) |
| `DCPOptimizedS3Reader` errors | Incompatible checkpoint format or access pattern | Use fallback reader (see [DCPOptimizedS3Reader Errors](#dcpoptimizeds3reader-errors)) |
| 503 Slow Down errors | S3 request throttling from concentrated key prefixes | Use prefix strategies (see [S3 Throttling](#s3-throttling)) |
| Segfault or hang in DataLoader workers | Fork safety issue with CRT threads | See [Fork Safety Issues](#fork-safety-issues) |
| "Connection reset" / timeout errors | Network instability or insufficient retries | Increase `max_attempts` in `S3ClientConfig` (see [Connection Issues](#connection-issues)) |
| OOM with `S3MapDataset` | Large object listing held in memory | Use `S3IterableDataset` for large datasets |

---

## Credentials Issues

The error "No signing credentials found" typically means AWS credentials are not configured or have expired.

**Checklist:**

1. Verify credentials are available: `aws sts get-caller-identity`
2. If using a credential URI (e.g., on ECS/EKS), ensure it has a trailing `/`
3. If using a named profile, set `AWS_PROFILE` or pass it to `S3ClientConfig(profile="my-profile")`
4. On EC2, verify the instance has an IAM role attached

```python
from s3torchconnector import S3ClientConfig

# Use a specific AWS profile
config = S3ClientConfig(profile="my-profile")
dataset = S3MapDataset.from_prefix(URI, region=REGION, s3client_config=config)
```

---

## DCPOptimizedS3Reader Errors

### Background

Starting in v1.5.0, `S3StorageReader` uses `DCPOptimizedS3Reader` by default for improved performance during distributed checkpoint loading. This reader requires sequential access patterns and range metadata injected by `S3StorageReader.prepare_local_plan()`.

If you encounter errors with the default reader, it may be due to:

- Non-standard checkpoint formats
- Custom `LoadPlanner` implementations that bypass `prepare_local_plan`
- Checkpoints saved with non-standard tools

### Fallback Solution

Use the sequential reader as a fallback:

```python
from s3torchconnector import S3ReaderConstructor
from s3torchconnector.dcp import S3StorageReader

storage_reader = S3StorageReader(
    region=REGION,
    path=CHECKPOINT_URI,
    reader_constructor=S3ReaderConstructor.sequential()
)
```

### When to Use Fallback vs. Fix Root Cause

- **Use fallback** if you have a custom `LoadPlanner` or non-standard checkpoint format that you cannot change.
- **Fix root cause** if you control the checkpoint save process — ensure you're using `S3StorageWriter` to save, which produces compatible checkpoints.

If you encounter errors with standard PyTorch DCP checkpoints, please [submit a GitHub issue](https://github.com/awslabs/s3-connector-for-pytorch/issues) describing your use case. We'd like to understand your scenario and potentially extend `DCPOptimizedS3Reader` to support it.

---

## S3 Throttling

### Symptom

503 Slow Down errors during distributed checkpoint save, especially with many ranks writing simultaneously.

### Cause

S3 partitions data by key prefix. When many processes write to the same prefix concurrently, requests concentrate on a single partition, triggering throttling.

### Solution

Use a prefix strategy to distribute writes across S3 partitions:

```python
from s3torchconnector.dcp import S3StorageWriter, HexPrefixStrategy

writer = S3StorageWriter(
    region=REGION,
    path=CHECKPOINT_URI,
    prefix_strategy=HexPrefixStrategy(epoch_num=1),
)
```

Available strategies: `HexPrefixStrategy`, `BinaryPrefixStrategy`, `RoundRobinPrefixStrategy`. See [Distributed Checkpoints — Prefix Strategies](DISTRIBUTED_CHECKPOINTS.md#s3-prefix-strategies) for details.

---

## Connection Issues

### Symptom

"Connection reset by peer", timeout errors, or intermittent failures.

### Solution

Increase the retry count via `S3ClientConfig`:

```python
from s3torchconnector import S3ClientConfig

config = S3ClientConfig(max_attempts=20)
dataset = S3MapDataset.from_prefix(URI, region=REGION, s3client_config=config)
```

If issues persist, enable [debug logging](#debug-logging) to inspect the underlying network errors.

---

## Performance Issues

### Slow Downloads

- **Check `throughput_target_gbps`**: The default is 10 Gbps. On larger EC2 instances (e.g., p4d, p5), increase this to match available network bandwidth:
  ```python
  config = S3ClientConfig(throughput_target_gbps=25)
  ```
- **Verify instance network capacity**: The CRT auto-tunes parallelism to hit the throughput target, but cannot exceed the instance's network limit.

### High Memory Usage

- **For datasets**: Switch from `S3MapDataset` (buffers entire object list) to `S3IterableDataset` for large datasets.
- **For reading large objects**: Use the range-based reader to avoid buffering entire objects:
  ```python
  from s3torchconnector import S3ReaderConstructor

  dataset = S3MapDataset.from_prefix(
      URI, region=REGION,
      reader_constructor=S3ReaderConstructor.range_based()
  )
  ```

### Slow DCP Loading

- **Ensure latest version**: v1.5.0+ uses `DCPOptimizedS3Reader` by default, which provides faster checkpoint loading than the sequential reader.
- **Increase IO threads for save**: Use `thread_count` in `S3StorageWriter`:
  ```python
  writer = S3StorageWriter(region=REGION, path=URI, thread_count=8)
  ```

---

## Fork Safety Issues

### Background

PyTorch's `DataLoader` with `num_workers > 0` uses `os.fork()` to create worker processes. The S3 Connector handles fork safety automatically by:

1. Detecting PID changes on every S3 client access
2. Registering `os.register_at_fork` handlers to clean up CRT threads before fork
3. Lazily reinitializing the native client in child processes

### If Issues Persist

If you experience segfaults or hangs related to multiprocessing:

1. **Update to the latest version** — fork safety improvements are ongoing.
2. **Switch to spawn start method** (avoids fork entirely):
   ```python
   import multiprocessing
   multiprocessing.set_start_method('spawn')
   ```
3. **Ensure `s3torchconnector` is imported in the main process** before the DataLoader creates workers — this registers the fork handlers.

---

## Still Stuck?

If none of the above resolves your issue:

1. Enable [debug logging](#debug-logging) and capture the output
2. [Open a GitHub issue](https://github.com/awslabs/s3-connector-for-pytorch/issues/new/choose) with the logs, your environment details, and a minimal reproduction
