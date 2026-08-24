"""Integration tests for the llmns extension module."""

import pytest

import llmns


def test_spec_examples_round_trip() -> None:
    examples = [
        "llms://api.anthropic.com/claude-fable-5",
        "llms://api.openai.com/gpt-5@version:2026-03-01",
        "llms://work@api.openai.com/gpt-5",
        "llms://huggingface.co/meta-llama/Llama-3.1-8B@hash:6f6073b",
        "llm://localhost:11434/llama3.2:3b?api=openai",
        "llms+grpc://triton.internal:8001/qwen3-ft@name:step-2000",
    ]
    for example in examples:
        assert str(llmns.parse(example)) == example


def test_full_reference_decomposes() -> None:
    ref = llmns.parse("llms+grpc://work@triton.internal:8001/qwen3-ft@name:step-2000?api=openai")
    assert ref.tls is True
    assert ref.transport == "grpc"
    assert ref.credential == "work"
    assert ref.host == "triton.internal"
    assert ref.port == 8001
    assert ref.model == "qwen3-ft"
    assert ref.pin == llmns.Pin("name", "step-2000")
    assert ref.hints == {"api": "openai"}


def test_transport_defaults_to_http() -> None:
    ref = llmns.parse("llm://localhost:11434/llama3.2:3b")
    assert ref.transport == "http"
    assert ref.tls is False
    assert ref.pin is None
    assert ref.hints == {}


def test_identity_ignores_credential_hints_transport_and_tls() -> None:
    a = llmns.parse("llms://work@API.openai.com/gpt-5?api=openai")
    b = llmns.parse("llm+grpc://api.openai.com/gpt-5")
    assert a == b
    assert hash(a) == hash(b)


def test_identity_uses_host_port_model_and_pin() -> None:
    base = llmns.parse("llm://localhost:8000/m")
    assert base != llmns.parse("llm://localhost:8001/m")
    assert base != llmns.parse("llm://localhost:8000/n")
    assert base != llmns.parse("llm://localhost:8000/m@hash:abc")
    assert llmns.parse("llm://localhost:8000/m@hash:abc") != llmns.parse("llm://localhost:8000/m@name:abc")


def test_references_key_dicts_by_model_identity() -> None:
    served = {llmns.parse("llms://api.openai.com/gpt-5"): "primary"}
    assert served[llmns.parse("llm://API.OPENAI.COM/gpt-5")] == "primary"


def test_normalized_lowercases_the_host_only() -> None:
    ref = llmns.parse("llms://API.OpenAI.com/GPT-5")
    assert str(ref.normalized()) == "llms://api.openai.com/GPT-5"


def test_constructor_builds_the_same_reference() -> None:
    built = llmns.Reference(
        "triton.internal",
        "qwen3-ft",
        transport="grpc",
        credential="work",
        port=8001,
        pin=llmns.Pin("name", "step-2000"),
        hints="api=openai",
    )
    parsed = llmns.parse("llms+grpc://work@triton.internal:8001/qwen3-ft@name:step-2000?api=openai")
    assert built == parsed
    assert str(built) == str(parsed)


@pytest.mark.parametrize(
    "bad",
    [
        "https://api.openai.com/gpt-5",
        "llmx://h/m",
        "llm://h",
        "llm:///m",
        "llm://h/",
        "llm://h/m@tag:x",
        "llm://h/m@name:",
        "llm://a@b@h/m",
        "llm://secret:hunter2@h/m",
        "llm://h:99999/m",
        "llm+://h/m",
    ],
)
def test_parse_errors(bad: str) -> None:
    with pytest.raises(llmns.ParseError):
        llmns.parse(bad)


def test_constructor_rejects_unencoded_at_in_model() -> None:
    with pytest.raises(llmns.ParseError):
        llmns.Reference("h", "model@with-at")


def test_version_is_exposed() -> None:
    assert llmns.__version__
