# Development

## Prerequisites

### Linux (Ubuntu/Debian)

```shell
sudo apt update && sudo apt install python3 python3-pip python3-venv clang cmake pkg-config
```

### macOS

```shell
brew install python3 cmake
xcode-select --install  # provides clang
```

## Getting Started

### Clone

```shell
git clone https://github.com/awslabs/s3-connector-for-pytorch.git
cd s3-connector-for-pytorch
```

### Virtual environment

```shell
python3 -m venv venv
source venv/bin/activate       # bash/zsh
source venv/bin/activate.fish  # fish
```

### Install Rust toolchain

```shell
curl https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"
```

The repo pins Rust to a specific version via `s3torchconnectorclient/rust-toolchain.toml`; rustup will auto-install it on first `cargo` invocation.

### Install in editable mode

```shell
pip install -e "s3torchconnectorclient[test]"
pip install -e "s3torchconnector[test]"
```

Additional extras for specific test suites:

- `s3torchconnector[dcp-test]` — Distributed Checkpoint tests
- `s3torchconnector[lightning-tests]` — PyTorch Lightning tests
- `s3torchconnector[e2e]` — End-to-end integration tests

### Rebuilding after Rust changes

When you modify Rust code, rebuild before running Python:

```shell
pip install -e s3torchconnectorclient
```

## Testing

### Unit tests

No AWS credentials required.

```shell
pytest s3torchconnectorclient/python/tst/unit
pytest s3torchconnector/tst/unit --ignore-glob='**/lightning/**' --ignore-glob='**/dcp/**'
```

**Lightning unit tests:**

```shell
pip install -e 's3torchconnector[test,lightning-tests]'
pytest s3torchconnector/tst/unit/lightning
```

**DCP unit tests:**

```shell
pip install -e 's3torchconnector[test,dcp-test]'
pytest s3torchconnector/tst/unit/dcp
```

### Integration / e2e tests

Require AWS credentials and an S3 bucket. Set the following environment variables:

| Variable | Description |
|----------|-------------|
| `CI_REGION` | AWS region (e.g. `us-east-1`) |
| `CI_BUCKET` | S3 bucket name |
| `CI_PREFIX` | Key prefix for test objects (must be empty or end with `/`) |
| `CI_STORAGE_CLASS` | Empty for Standard, `EXPRESS_ONEZONE` for directory buckets |

```shell
pip install -e 's3torchconnector[test,e2e]'
pytest s3torchconnectorclient/python/tst/integration
pytest s3torchconnector/tst/e2e
```

### Running tests in parallel

```shell
pip install pytest-xdist
pytest -n auto
```

> **Note:** Do not use `-n auto` for distributed training tests — concurrent runs cause port clashes.

## Linting

### Rust

From the repo root:

```shell
cargo clippy --all-targets --all-features --manifest-path s3torchconnectorclient/Cargo.toml
```

### Python

```shell
black --verbose .
flake8 s3torchconnector/ --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 s3torchconnectorclient/python --count --select=E9,F63,F7,F82 --show-source --statistics
mypy s3torchconnector/src
mypy s3torchconnectorclient/python/src
```

To run mypy without Lightning installed:

```shell
mypy s3torchconnector/src --exclude s3torchconnector/src/s3torchconnector/lightning
mypy s3torchconnectorclient/python/src
```

## Debugging

### Rust (GDB)

Run the Rust test in question with the GDB debugger enabled directly.

### Python + Rust (GDB)

Create a "Custom Build Application" run configuration. Set the executable path to your virtual environment's Python binary (`venv/bin/python`) and the script name as the program argument. Place breakpoints in Rust/C code and run.

### Runtime logging

For runtime debug logging (Rust/CRT log levels, log file output), see [Troubleshooting — Debug Logging](docs/TROUBLESHOOTING.md#debug-logging).

## Updating mountpoint-s3-client

1. Edit the version in `s3torchconnectorclient/Cargo.toml`
2. Run `cargo clippy --all-targets --all-features --manifest-path s3torchconnectorclient/Cargo.toml` to update `Cargo.lock`
3. Rebuild: `pip install -e s3torchconnectorclient`
4. Run unit tests to validate

## Building Wheels Locally

Uses [`cibuildwheel`](https://cibuildwheel.readthedocs.io/). Requires Docker on Linux (or Finch on macOS).

```shell
pip install cibuildwheel
cibuildwheel --only cp312-manylinux_x86_64 s3torchconnectorclient
```

See `.github/workflows/wheels.yml` for the full CI wheel build configuration.

## Licensing

Ensure all new files have a copyright header:

> Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
>
> // SPDX-License-Identifier: BSD

**IDE setup (PyCharm/CLion):** Go to Settings → Copyright Profiles, create a profile with the text above. Then under Copyright, create a scope covering "All" and assign the profile.

## Configuration

For runtime configuration (throughput targets, part size, unsigned requests), see [Configuration](docs/CONFIGURATION.md).

## Further Reading

- [Configuration](docs/CONFIGURATION.md) — S3 client tuning, credentials, S3 Express
- [Troubleshooting](docs/TROUBLESHOOTING.md) — Debug logging, common issues
- [Benchmarking](s3torchbenchmarking/README.md) — Performance testing
- [Contributing](CONTRIBUTING.md) — How to submit changes
