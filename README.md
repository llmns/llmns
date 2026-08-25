# llmns

Parse, normalize, and compare `llm://` references, per
[llmns draft v02](https://llmns.info/rfc.html).

A reference identifies a language model by serving host, transport,
model state, and credential name:

```
llm[s][+transport]://[credential@]host[:port]/model[@pin][?hints]
```

## Install

```
pip install llmns
```

## Use

```python
import llmns

ref = llmns.parse("llms+grpc://work@triton.internal:8001/qwen3-ft@name:step-2000?api=openai")
ref.host        # "triton.internal"
ref.port        # 8001
ref.tls         # True
ref.transport   # "grpc"
ref.credential  # "work" — a name in the client's credential store, never the secret
ref.model       # "qwen3-ft"
ref.pin         # Pin(kind="name", value="step-2000")
ref.hints       # {"api": "openai"}

# Equality is the specification's equivalence rule: the normalized
# (host, port, model, pin) tuple, with RFC 3986 percent-encoding
# normalization. Credential, hints, transport, and TLS do not contribute.
llmns.parse("llms://work@API.openai.com/gpt-5") == llmns.parse("llm://api%2Eopenai.com/gpt-5")  # True
```

The core is Rust; the wheel is built with [maturin](https://maturin.rs).

## License

MIT OR Apache-2.0
