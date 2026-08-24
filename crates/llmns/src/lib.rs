//! Parse, normalize, and compare llm:// references (llmns draft v01).
//!
//! ```
//! let reference: llmns::Reference =
//!     "llms+grpc://work@triton.internal:8001/qwen3-ft@name:step-2000".parse()?;
//! assert_eq!(reference.host, "triton.internal");
//! assert!(reference.tls);
//! # Ok::<(), llmns::ParseError>(())
//! ```

#![forbid(unsafe_code)]

use std::fmt;
use std::str::FromStr;

/// An invalid reference, pin, or component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

/// How firmly a pin fixes the model state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PinKind {
    /// A symbolic reference the provider may move.
    Name,
    /// A content or commit hash; immutable.
    Hash,
    /// A label the provider publishes once and never reuses.
    Version,
}

impl PinKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::Hash => "hash",
            Self::Version => "version",
        }
    }
}

impl FromStr for PinKind {
    type Err = ParseError;

    fn from_str(kind: &str) -> Result<Self, ParseError> {
        match kind {
            "name" => Ok(Self::Name),
            "hash" => Ok(Self::Hash),
            "version" => Ok(Self::Version),
            _ => Err(ParseError(format!(
                "pin kind must be \"name\", \"hash\", or \"version\", not {kind:?}"
            ))),
        }
    }
}

impl fmt::Display for PinKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A typed pointer to one fixed state of a model.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Pin {
    pub kind: PinKind,
    pub value: String,
}

impl Pin {
    pub fn new(kind: PinKind, value: impl Into<String>) -> Result<Self, ParseError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ParseError("empty pin value".to_string()));
        }
        Ok(Pin { kind, value })
    }
}

impl fmt::Display for Pin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind, self.value)
    }
}

/// The normalized (host, model, pin) triple that decides whether two
/// references denote the same model. The credential, the hints, the
/// transport, and TLS do not contribute.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Identity {
    pub host: String,
    pub port: Option<u16>,
    pub model: String,
    pub pin: Option<Pin>,
}

/// One llm:// reference, decomposed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reference {
    pub tls: bool,
    /// The transport marker; `None` means HTTP.
    pub transport: Option<String>,
    pub credential: Option<String>,
    pub host: String,
    pub port: Option<u16>,
    pub model: String,
    pub pin: Option<Pin>,
    /// The raw hints, everything after "?".
    pub query: Option<String>,
}

fn checked_transport(transport: &str) -> Result<String, ParseError> {
    if transport.is_empty() || !transport.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(ParseError(format!(
            "transport must be one or more ASCII letters or digits, not {transport:?}"
        )));
    }
    Ok(transport.to_string())
}

fn checked_port(port: &str) -> Result<u16, ParseError> {
    port.parse::<u16>()
        .map_err(|_| ParseError(format!("invalid port {port:?}")))
}

impl Reference {
    /// Build a reference from components, applying the same validation as
    /// parsing.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        host: &str,
        model: &str,
        tls: bool,
        transport: Option<&str>,
        credential: Option<&str>,
        port: Option<u16>,
        pin: Option<Pin>,
        query: Option<&str>,
    ) -> Result<Self, ParseError> {
        if host.is_empty() || host.contains(['@', '/', '?']) {
            return Err(ParseError(format!("invalid host {host:?}")));
        }
        if model.is_empty() || model.contains(['@', '?']) {
            return Err(ParseError(format!(
                "invalid model {model:?}; percent-encode a literal \"@\" as %40"
            )));
        }
        if let Some(credential) = credential {
            if credential.is_empty() || credential.contains(['@', '/', '?', ':']) {
                return Err(ParseError(format!(
                    "invalid credential name {credential:?}"
                )));
            }
        }
        let transport = match transport {
            Some(transport) => Some(checked_transport(transport)?),
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
            query: query.map(str::to_string),
        })
    }

    /// The transport, "http" when the reference names none.
    pub fn transport_or_default(&self) -> &str {
        self.transport.as_deref().unwrap_or("http")
    }

    /// The normalized (host, model, pin) triple.
    pub fn identity(&self) -> Identity {
        Identity {
            host: self.host.to_ascii_lowercase(),
            port: self.port,
            model: self.model.clone(),
            pin: self.pin.clone(),
        }
    }

    /// Whether both references denote the same model per the identity rule.
    pub fn denotes_same_model(&self, other: &Reference) -> bool {
        self.identity() == other.identity()
    }

    /// The reference with the host lowercased; everything else is unchanged.
    pub fn normalized(&self) -> Reference {
        let mut normalized = self.clone();
        normalized.host = normalized.host.to_ascii_lowercase();
        normalized
    }

    /// The hints as key/value pairs; a pair without "=" yields an empty value.
    pub fn hints(&self) -> impl Iterator<Item = (&str, &str)> {
        self.query
            .as_deref()
            .unwrap_or("")
            .split('&')
            .filter(|pair| !pair.is_empty())
            .map(|pair| pair.split_once('=').unwrap_or((pair, "")))
    }
}

impl FromStr for Reference {
    type Err = ParseError;

    fn from_str(reference: &str) -> Result<Self, ParseError> {
        let rest = reference
            .strip_prefix("llm")
            .ok_or_else(|| ParseError("the scheme must start with \"llm\"".to_string()))?;
        let separator = rest
            .find("://")
            .ok_or_else(|| ParseError("missing \"://\"".to_string()))?;
        let (scheme_rest, after) = (&rest[..separator], &rest[separator + 3..]);
        let (tls, marker) = match scheme_rest.strip_prefix('s') {
            Some(marker) => (true, marker),
            None => (false, scheme_rest),
        };
        let transport = match marker.strip_prefix('+') {
            Some(transport) => Some(checked_transport(transport)?),
            None if marker.is_empty() => None,
            None => return Err(ParseError(format!("unexpected scheme suffix {marker:?}"))),
        };

        let slash = after
            .find('/')
            .ok_or_else(|| ParseError("missing \"/model\"".to_string()))?;
        let (authority, path) = (&after[..slash], &after[slash + 1..]);
        let (credential, hostport) = match authority.split_once('@') {
            Some((credential, hostport)) => {
                if credential.is_empty() {
                    return Err(ParseError("empty credential name".to_string()));
                }
                if credential.contains(':') {
                    return Err(ParseError(
                        "the credential is a name; a reference never carries a secret".to_string(),
                    ));
                }
                (Some(credential.to_string()), hostport)
            }
            None => (None, authority),
        };
        if hostport.contains('@') {
            return Err(ParseError(
                "more than one \"@\" in the authority".to_string(),
            ));
        }
        let (host, port) = if hostport.starts_with('[') {
            let end = hostport
                .find(']')
                .ok_or_else(|| ParseError("unterminated IPv6 literal".to_string()))?;
            let (host, rest) = (&hostport[..=end], &hostport[end + 1..]);
            match rest.strip_prefix(':') {
                Some(port) => (host, Some(checked_port(port)?)),
                None if rest.is_empty() => (host, None),
                None => {
                    return Err(ParseError(format!(
                        "unexpected characters after IPv6 literal: {rest:?}"
                    )))
                }
            }
        } else {
            match hostport.rsplit_once(':') {
                Some((host, port)) => (host, Some(checked_port(port)?)),
                None => (hostport, None),
            }
        };
        if host.is_empty() {
            return Err(ParseError("empty host".to_string()));
        }

        let (path, query) = match path.split_once('?') {
            Some((path, query)) => (path, Some(query.to_string())),
            None => (path, None),
        };
        let (model, pin) = match path.split_once('@') {
            Some((model, pin)) => {
                let (kind, value) = pin.split_once(':').ok_or_else(|| {
                    ParseError("a pin is a kind, a colon, and a value".to_string())
                })?;
                (model, Some(Pin::new(kind.parse()?, value)?))
            }
            None => (path, None),
        };
        if model.is_empty() {
            return Err(ParseError("empty model".to_string()));
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
}

impl fmt::Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("llm")?;
        if self.tls {
            f.write_str("s")?;
        }
        if let Some(transport) = &self.transport {
            write!(f, "+{transport}")?;
        }
        f.write_str("://")?;
        if let Some(credential) = &self.credential {
            write!(f, "{credential}@")?;
        }
        f.write_str(&self.host)?;
        if let Some(port) = self.port {
            write!(f, ":{port}")?;
        }
        write!(f, "/{}", self.model)?;
        if let Some(pin) = &self.pin {
            write!(f, "@{pin}")?;
        }
        if let Some(query) = &self.query {
            write!(f, "?{query}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{PinKind, Reference};

    fn parse(reference: &str) -> Reference {
        reference.parse().unwrap()
    }

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
            assert_eq!(parse(example).to_string(), example);
        }
    }

    #[test]
    fn full_reference_decomposes() {
        let r = parse("llms+grpc://work@triton.internal:8001/qwen3-ft@name:step-2000?api=openai");
        assert!(r.tls);
        assert_eq!(r.transport.as_deref(), Some("grpc"));
        assert_eq!(r.credential.as_deref(), Some("work"));
        assert_eq!(r.host, "triton.internal");
        assert_eq!(r.port, Some(8001));
        assert_eq!(r.model, "qwen3-ft");
        let pin = r.pin.as_ref().unwrap();
        assert_eq!((pin.kind, pin.value.as_str()), (PinKind::Name, "step-2000"));
        assert_eq!(r.hints().collect::<Vec<_>>(), vec![("api", "openai")]);
    }

    #[test]
    fn model_keeps_colons_and_slashes() {
        assert_eq!(
            parse("llm://localhost:11434/llama3.2:3b").model,
            "llama3.2:3b"
        );
        assert_eq!(
            parse("llms://huggingface.co/meta-llama/Llama-3.1-8B").model,
            "meta-llama/Llama-3.1-8B"
        );
    }

    #[test]
    fn identity_ignores_credential_hints_transport_and_tls() {
        let a = parse("llms://work@API.openai.com/gpt-5?api=openai");
        let b = parse("llm+grpc://api.openai.com/gpt-5");
        assert!(a.denotes_same_model(&b));
    }

    #[test]
    fn identity_uses_port_model_and_pin() {
        let a = parse("llm://localhost:8000/m");
        assert!(!a.denotes_same_model(&parse("llm://localhost:8001/m")));
        assert!(!a.denotes_same_model(&parse("llm://localhost:8000/n")));
        let pinned = parse("llm://localhost:8000/m@hash:abc");
        assert!(!a.denotes_same_model(&pinned));
        assert!(!pinned.denotes_same_model(&parse("llm://localhost:8000/m@name:abc")));
    }

    #[test]
    fn ipv6_literals() {
        let r = parse("llm://[::1]:8000/m");
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
            bad.parse::<Reference>().expect_err(bad);
        }
    }
}
