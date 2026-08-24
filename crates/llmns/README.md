# llmns

Parse, normalize, and compare `llm://` references, per
[draft-tahrioui-llmns-01](https://llmns.info/rfc.html).

```
llm[s][+transport]://[credential@]host[:port]/model[@pin][?hints]
```

```rust
let reference: llmns::Reference =
    "llms+grpc://work@triton.internal:8001/qwen3-ft@name:step-2000".parse()?;
assert_eq!(reference.host, "triton.internal");
assert_eq!(reference.pin.as_ref().unwrap().kind, llmns::PinKind::Name);

// Identity is the specification's rule: the normalized (host, model, pin)
// triple. Credential, hints, transport, and TLS do not contribute.
let other: llmns::Reference = "llm://TRITON.internal:8001/qwen3-ft@name:step-2000".parse()?;
assert!(reference.denotes_same_model(&other));
# Ok::<(), llmns::ParseError>(())
```

Python bindings ship on PyPI as [`llmns`](https://pypi.org/project/llmns/).

## License

MIT OR Apache-2.0
