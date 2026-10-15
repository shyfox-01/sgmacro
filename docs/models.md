# Model setup

The saved report works without a model:

```bash
sgmacro run --as-of 2026-10-02 --offline --llm cache
```

Discovery needs a model with tool support. A fresh report can also use a model
to draft its summary; `--llm off` selects the template instead.

## Backends

`sgmacro models` lists available backends and explains their configuration.
Listing a model does not verify that generation will work.

| Backend | Configuration |
| --- | --- |
| Ollama | Run a local Ollama server with a tool-capable model; optionally set `SGMACRO_OLLAMA_MODEL` |
| OpenAI-compatible | Set `SGMACRO_OPENAI_BASE_URL`, `SGMACRO_OPENAI_MODEL` and the appropriate key |
| Anthropic | Install the optional `anthropic` extra and set `ANTHROPIC_API_KEY`; the backend's model default is defined in `src/sgmacro/llm/backends.py` |

Automatic resolution checks Anthropic, the configured OpenAI-compatible service,
then Ollama. Explicit backend selection avoids depending on that order.

### Ollama

The recorded discovery runs used `qwen3.5:9b`. The bundled summary was generated
with `qwen3.5:27b`. Model downloads and inference are separate from installing
this Python package.

With a suitable model already installed and Ollama running:

```powershell
$env:SGMACRO_OLLAMA_MODEL = "qwen3.5:9b"
sgmacro models
sgmacro discover --brief predictors --backend ollama
```

`SGMACRO_OLLAMA_URL` changes the server address. Offline mode permits a local
server but rejects a remote Ollama address.

### OpenAI-compatible services

Supported base-URL presets are `openai`, `gemini`, `deepseek`, `qwen` and
`openrouter`. A custom API base URL is also accepted.

| Preset | Provider key variable |
| --- | --- |
| `openai` | `OPENAI_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `qwen` | `DASHSCOPE_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |

A custom address uses `SGMACRO_OPENAI_API_KEY`, not a different provider's key.
Remote authenticated requests require HTTPS.

Example for an already running local OpenAI-compatible server:

```powershell
$env:SGMACRO_OPENAI_BASE_URL = "http://127.0.0.1:1234/v1"
$env:SGMACRO_OPENAI_MODEL = "YOUR_INSTALLED_MODEL"
sgmacro models
sgmacro discover --brief predictors --backend openai
```

Use your server's address and model identifier. No server is started by these
commands.

For a remote service, supply its key through the process environment rather than
a command-line argument. The backend supports a JSON-object fallback when a
service rejects JSON-schema output. Authentication failures, rate limits and
server failures remain errors.

## Limits and recovery

A discovery session has a default budget of forty model calls
(`--max-steps`). The predictor brief allows at most four proposals; repair allows
two. These are limits, not promised result counts.

Identical effective tool calls reuse stored results. A failed call can be retried
once when requested again. Three consecutive steps consisting only of
unproductive repeats stop the session. Near two thirds of the step budget, the
agent is prompted to finish with the candidates it has.

For the native Ollama backend, the request uses a 32k context window. Older tool
results are shortened while the latest six are retained in full. The full
session remains on disk. Estimated context use triggers a wrap-up prompt at 60%
and a stop at 75%. These are margins based on token counts and estimates,
not exact guarantees.

A failed or truncated model response ends the session with an error while
preserving completed steps and proposals for verification.

Local requests use temperature zero and a fixed seed. That reduces variability
but does not guarantee identical new generations. Model versions, backend
behavior, inputs and search history also affect the result. Cached replay is the
reproducible path.

## Data sent to models

The summary request contains computed facts and signals. Discovery requests
contain the search brief and tool results. When using a remote provider, these
inputs are sent to that service.

Credentials are read from environment variables. Error text and saved sessions
are redacted, and report, manifest and cache writes include credential checks.
Keep keys out of custom briefs and source files.

## What has been tested

The recorded native Ollama sessions and the OpenAI-compatible discovery session
used local Qwen models. The latter produced four verified proposals in twenty
steps. Its summary request timed out; native Ollama disables thinking for these
requests.

Anthropic and hosted OpenAI-compatible services were tested against simulated
servers. Their live endpoints have not been verified by this project.

The [summary comparison](evidence/summary_models.md) uses one fact sheet and fresh
caches. Qwen 9B passed eight of nine claims and 27B passed nine of nine; the rejected
9B claim exceeded the length limit. This does not establish a general ranking
of their reasoning or discovery performance.
