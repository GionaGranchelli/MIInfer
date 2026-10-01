# MIInfer

MIInfer is a focused LLM runtime for AMD gfx906 GPUs, qualified on the AMD
Instinct MI50 32 GB. The CLI selects the qualified production profile
automatically; normal use does not require MIInfer environment variables or
knowledge of kernel presets.

## Install and start

On a supported Linux/MI50 system, download and extract the release archive,
run its installer, then add the installed binary directory to `PATH`:

```bash
tar -xzf miinfer-0.2.0-gfx906-Linux.tar.gz
./miinfer-0.2.0-gfx906-Linux/install.sh miinfer-0.2.0-gfx906-Linux.tar.gz
export PATH="$HOME/.local/miinfer/bin:$PATH"
miinfer doctor --model ~/models/Qwen3.8-27B-Q4_K_M.gguf
miinfer chat ~/models/Qwen3.8-27B-Q4_K_M.gguf
```

For a one-shot response:

```bash
miinfer run ~/models/Qwen3.8-27B-Q4_K_M.gguf "Hello"
```

Start the OpenAI-compatible API and bundled web UI on localhost:

```bash
miinfer serve ~/models/Qwen3.8-27B-Q4_K_M.gguf
```

The API is available at `http://127.0.0.1:8080/v1`; the web UI is at
`http://127.0.0.1:8080/`. For LAN access, create a key file and bind explicitly:

```bash
mkdir -p ~/.config/miinfer
openssl rand -hex 32 > ~/.config/miinfer/api-key
chmod 600 ~/.config/miinfer/api-key
miinfer serve ~/models/Qwen3.8-27B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 --api-key-file ~/.config/miinfer/api-key
```

Set stable defaults once with `miinfer config set model-dir ~/models` and
`miinfer config set default-model Qwen3.8-27B-Q4_K_M.gguf`. Then `miinfer chat`
and `miinfer serve` can omit the model argument. Run `miinfer <command> --help`
for concise command options; `doctor`, `models`, `inspect`, and `config` also
support `--json` where applicable.

## Qualification

The MI50 qualification measured MIInfer ahead of pinned mx-llama.cpp and
upstream llama.cpp in all nine tested prefill/decode cells. The claim is
specific to the qualified model, hardware, and workload matrix. Decode used
forced-token model-forward replay, not HTTP generation throughput; the
multi-turn serving workload passed with prefix reuse but was not compared
against the other runtimes. See [release notes and qualification scope](docs/release.md).

The P8192 prefill lead over mx is narrow. The isolated attention kernel remains
slower, but attention research is deferred because the complete qualified
runtime wins the measured system-level matrix.

For source builds, installation details, hardware support, and the public CLI
contract, see [`docs/release.md`](docs/release.md),
[`docs/hardware.md`](docs/hardware.md), and
[`docs/cli-product-contract.md`](docs/cli-product-contract.md).
