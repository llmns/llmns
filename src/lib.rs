//! llmns: parse, normalize, and compare llm:// references (draft-tahrioui-llmns-01).

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

create_exception!(_llmns, ParseError, PyValueError, "Invalid llmns reference.");

const PIN_KINDS: [&str; 3] = ["name", "hash", "version"];

fn checked_pin(kind: &str, value: &str) -> Result<Pin, String> {
    if !PIN_KINDS.contains(&kind) {
        return Err(format!(
            "pin kind must be \"name\", \"hash\", or \"version\", not {kind:?}"
        ));
    }
    if value.is_empty() {
        return Err("empty pin value".to_string());
    }
    Ok(Pin {
        kind: kind.to_string(),
        value: value.to_string(),
    })
}

/// A typed pointer to one fixed state of a model.
#[pyclass(frozen)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Pin {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub value: String,
}

#[pymethods]
impl Pin {
    #[new]
    fn new(kind: &str, value: &str) -> PyResult<Self> {
        checked_pin(kind, value).map_err(ParseError::new_err)
    }

    fn __str__(&self) -> String {
        format!("{}:{}", self.kind, self.value)
    }

    fn __repr__(&self) -> String {
        format!("Pin(kind={:?}, value={:?})", self.kind, self.value)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<Pin>() {
            Ok(o) => *self == o,
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// One llm:// reference, decomposed.
///
/// Equality and hashing follow the specification's identity rule: two
/// references denote the same model when their normalized
/// (host, model, pin) triples are equal. The credential, the hints, the
/// transport, and TLS do not contribute.
#[pyclass(frozen)]
#[derive(Clone, Debug)]
pub struct Reference {
    #[pyo3(get)]
    pub tls: bool,
    transport: Option<String>,
    #[pyo3(get)]
    pub credential: Option<String>,
    #[pyo3(get)]
    pub host: String,
    #[pyo3(get)]
    pub port: Option<u16>,
    #[pyo3(get)]
    pub model: String,
    pin: Option<Pin>,
    query: Option<String>,
}

impl Reference {
    fn identity(&self) -> (String, Option<u16>, &str, Option<&Pin>) {
        (
            self.host.to_ascii_lowercase(),
            self.port,
            self.model.as_str(),
            self.pin.as_ref(),
        )
    }

    fn format(&self) -> String {
        let mut out = String::from("llm");
        if self.tls {
            out.push('s');
        }
        if let Some(transport) = &self.transport {
            out.push('+');
            out.push_str(transport);
        }
        out.push_str("://");
        if let Some(credential) = &self.credential {
            out.push_str(credential);
            out.push('@');
        }
        out.push_str(&self.host);
        if let Some(port) = self.port {
            out.push(':');
            out.push_str(&port.to_string());
        }
        out.push('/');
        out.push_str(&self.model);
        if let Some(pin) = &self.pin {
            out.push('@');
            out.push_str(&pin.kind);
            out.push(':');
            out.push_str(&pin.value);
        }
        if let Some(query) = &self.query {
            out.push('?');
            out.push_str(query);
        }
        out
    }
}

fn checked_transport(transport: &str) -> Result<String, String> {
    if transport.is_empty() || !transport.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(format!(
            "transport must be one or more ASCII letters or digits, not {transport:?}"
        ));
    }
    Ok(transport.to_string())
}

fn checked_port(port: &str) -> Result<u16, String> {
    port.parse::<u16>()
        .map_err(|_| format!("invalid port {port:?}"))
}

fn parse_reference(reference: &str) -> Result<Reference, String> {
    let rest = reference
        .strip_prefix("llm")
        .ok_or("the scheme must start with \"llm\"")?;
    let separator = rest.find("://").ok_or("missing \"://\"")?;
    let (scheme_rest, after) = (&rest[..separator], &rest[separator + 3..]);
    let (tls, marker) = match scheme_rest.strip_prefix('s') {
        Some(marker) => (true, marker),
        None => (false, scheme_rest),
    };
    let transport = match marker.strip_prefix('+') {
        Some(transport) => Some(checked_transport(transport)?),
        None if marker.is_empty() => None,
        None => return Err(format!("unexpected scheme suffix {marker:?}")),
    };

    let slash = after.find('/').ok_or("missing \"/model\"")?;
    let (authority, path) = (&after[..slash], &after[slash + 1..]);
    let (credential, hostport) = match authority.split_once('@') {
        Some((credential, hostport)) => {
            if credential.is_empty() {
                return Err("empty credential name".to_string());
            }
            if credential.contains(':') {
                return Err(
                    "the credential is a name; a reference never carries a secret".to_string(),
                );
            }
            (Some(credential.to_string()), hostport)
        }
        None => (None, authority),
    };
    if hostport.contains('@') {
        return Err("more than one \"@\" in the authority".to_string());
    }
    let (host, port) = if hostport.starts_with('[') {
        let end = hostport.find(']').ok_or("unterminated IPv6 literal")?;
        let (host, rest) = (&hostport[..=end], &hostport[end + 1..]);
        match rest.strip_prefix(':') {
            Some(port) => (host, Some(checked_port(port)?)),
            None if rest.is_empty() => (host, None),
            None => {
                return Err(format!(
                    "unexpected characters after IPv6 literal: {rest:?}"
                ))
            }
        }
    } else {
        match hostport.rsplit_once(':') {
            Some((host, port)) => (host, Some(checked_port(port)?)),
            None => (hostport, None),
        }
    };
    if host.is_empty() {
        return Err("empty host".to_string());
    }

    let (path, query) = match path.split_once('?') {
        Some((path, query)) => (path, Some(query.to_string())),
        None => (path, None),
    };
    let (model, pin) = match path.split_once('@') {
        Some((model, pin)) => {
            let (kind, value) = pin
                .split_once(':')
                .ok_or("a pin is a kind, a colon, and a value")?;
            (model, Some(checked_pin(kind, value)?))
        }
        None => (path, None),
    };
    if model.is_empty() {
        return Err("empty model".to_string());
    }

    Ok(Reference {
        tls,
        transport,
        credential,
        host: host.to_string(),
        port,
        model: model.to_string(),
        pin,
        query,
    })
}

#[pymethods]
impl Reference {
    #[new]
    #[pyo3(signature = (host, model, *, tls = true, transport = None, credential = None, port = None, pin = None, hints = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        host: &str,
        model: &str,
        tls: bool,
        transport: Option<&str>,
        credential: Option<&str>,
        port: Option<u16>,
        pin: Option<Pin>,
        hints: Option<&str>,
    ) -> PyResult<Self> {
        if host.is_empty() || host.contains(['@', '/', '?']) {
            return Err(ParseError::new_err(format!("invalid host {host:?}")));
        }
        if model.is_empty() || model.contains(['@', '?']) {
            return Err(ParseError::new_err(format!(
                "invalid model {model:?}; percent-encode a literal \"@\" as %40"
            )));
        }
        if let Some(credential) = credential {
            if credential.is_empty() || credential.contains(['@', '/', '?', ':']) {
                return Err(ParseError::new_err(format!(
                    "invalid credential name {credential:?}"
                )));
            }
        }
        let transport = match transport {
            Some(transport) => Some(checked_transport(transport).map_err(ParseError::new_err)?),
            None => None,
        };
        Ok(Reference {
            tls,
            transport,
            credential: credential.map(str::to_string),
            host: host.to_string(),
            port,
            model: model.to_string(),
            pin,
            query: hints.map(str::to_string),
        })
    }

    /// Parse a reference string.
    #[staticmethod]
    fn parse(reference: &str) -> PyResult<Self> {
        parse_reference(reference).map_err(|e| ParseError::new_err(format!("{e}: {reference:?}")))
    }

    #[getter]
    fn transport(&self) -> String {
        self.transport.clone().unwrap_or_else(|| "http".to_string())
    }

    #[getter]
    fn pin(&self) -> Option<Pin> {
        self.pin.clone()
    }

    #[getter]
    fn hints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let hints = PyDict::new(py);
        let Some(query) = &self.query else {
            return Ok(hints);
        };
        for pair in query.split('&').filter(|pair| !pair.is_empty()) {
            match pair.split_once('=') {
                Some((key, value)) => hints.set_item(key, value)?,
                None => hints.set_item(pair, "")?,
            }
        }
        Ok(hints)
    }

    /// The reference with the host lowercased; everything else is unchanged.
    fn normalized(&self) -> Reference {
        let mut normalized = self.clone();
        normalized.host = normalized.host.to_ascii_lowercase();
        normalized
    }

    fn __str__(&self) -> String {
        self.format()
    }

    fn __repr__(&self) -> String {
        format!("Reference({:?})", self.format())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<Reference>() {
            Ok(o) => self.identity() == o.identity(),
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.identity().hash(&mut hasher);
        hasher.finish()
    }
}

/// Parse a reference string.
#[pyfunction]
fn parse(reference: &str) -> PyResult<Reference> {
    Reference::parse(reference)
}

#[pymodule]
fn _llmns(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("ParseError", py.get_type::<ParseError>())?;
    m.add_class::<Pin>()?;
    m.add_class::<Reference>()?;
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_reference;

    #[test]
    fn spec_examples_round_trip() {
        let examples = [
            "llms://api.anthropic.com/claude-fable-5",
            "llms://api.openai.com/gpt-5@version:2026-03-01",
            "llms://work@api.openai.com/gpt-5",
            "llms://huggingface.co/meta-llama/Llama-3.1-8B@hash:6f6073b",
            "llm://localhost:11434/llama3.2:3b?api=openai",
            "llms+grpc://triton.internal:8001/qwen3-ft@name:step-2000",
        ];
        for example in examples {
            let reference = parse_reference(example).unwrap();
            assert_eq!(reference.format(), example);
        }
    }

    #[test]
    fn full_reference_decomposes() {
        let r = parse_reference(
            "llms+grpc://work@triton.internal:8001/qwen3-ft@name:step-2000?api=openai",
        )
        .unwrap();
        assert!(r.tls);
        assert_eq!(r.transport.as_deref(), Some("grpc"));
        assert_eq!(r.credential.as_deref(), Some("work"));
        assert_eq!(r.host, "triton.internal");
        assert_eq!(r.port, Some(8001));
        assert_eq!(r.model, "qwen3-ft");
        let pin = r.pin.as_ref().unwrap();
        assert_eq!(
            (pin.kind.as_str(), pin.value.as_str()),
            ("name", "step-2000")
        );
        assert_eq!(r.query.as_deref(), Some("api=openai"));
    }

    #[test]
    fn model_keeps_colons_and_slashes() {
        let r = parse_reference("llm://localhost:11434/llama3.2:3b").unwrap();
        assert_eq!(r.model, "llama3.2:3b");
        let r = parse_reference("llms://huggingface.co/meta-llama/Llama-3.1-8B").unwrap();
        assert_eq!(r.model, "meta-llama/Llama-3.1-8B");
    }

    #[test]
    fn identity_ignores_credential_hints_transport_and_tls() {
        let a = parse_reference("llms://work@API.openai.com/gpt-5?api=openai").unwrap();
        let b = parse_reference("llm+grpc://api.openai.com/gpt-5").unwrap();
        assert_eq!(a.identity(), b.identity());
    }

    #[test]
    fn identity_uses_port_model_and_pin() {
        let a = parse_reference("llm://localhost:8000/m").unwrap();
        assert_ne!(
            a.identity(),
            parse_reference("llm://localhost:8001/m")
                .unwrap()
                .identity()
        );
        assert_ne!(
            a.identity(),
            parse_reference("llm://localhost:8000/n")
                .unwrap()
                .identity()
        );
        let pinned = parse_reference("llm://localhost:8000/m@hash:abc").unwrap();
        assert_ne!(a.identity(), pinned.identity());
        let name_pin = parse_reference("llm://localhost:8000/m@name:abc").unwrap();
        assert_ne!(pinned.identity(), name_pin.identity());
    }

    #[test]
    fn ipv6_literals() {
        let r = parse_reference("llm://[::1]:8000/m").unwrap();
        assert_eq!(r.host, "[::1]");
        assert_eq!(r.port, Some(8000));
    }

    #[test]
    fn errors() {
        for bad in [
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
        ] {
            parse_reference(bad).expect_err(bad);
        }
    }
}
