//! Python bindings for llmns (llmns draft v02).

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

create_exception!(_llmns, ParseError, PyValueError, "Invalid llmns reference.");

fn parse_error(error: llmns::ParseError) -> PyErr {
    ParseError::new_err(error.0)
}

/// A typed pointer to one fixed state of a model.
#[pyclass(frozen, name = "Pin")]
#[derive(Clone)]
pub struct PyPin {
    inner: llmns::Pin,
}

#[pymethods]
impl PyPin {
    #[new]
    fn new(kind: &str, value: &str) -> PyResult<Self> {
        let kind: llmns::PinKind = kind.parse().map_err(parse_error)?;
        let inner = llmns::Pin::new(kind, value).map_err(parse_error)?;
        Ok(PyPin { inner })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind.as_str()
    }

    #[getter]
    fn value(&self) -> &str {
        &self.inner.value
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "Pin(kind={:?}, value={:?})",
            self.inner.kind.as_str(),
            self.inner.value
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyPin>() {
            Ok(o) => self.inner == o.inner,
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

/// One llm:// reference, decomposed.
///
/// Equality and hashing follow the specification's equivalence rule: two
/// references are equivalent when their normalized
/// (host, port, model, pin) tuples are equal. The credential, the hints,
/// the transport, and TLS do not contribute.
#[pyclass(frozen, name = "Reference")]
#[derive(Clone)]
pub struct PyReference {
    inner: llmns::Reference,
}

#[pymethods]
impl PyReference {
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
        pin: Option<PyPin>,
        hints: Option<&str>,
    ) -> PyResult<Self> {
        let inner = llmns::Reference::new(
            host,
            model,
            tls,
            transport,
            credential,
            port,
            pin.map(|pin| pin.inner),
            hints,
        )
        .map_err(parse_error)?;
        Ok(PyReference { inner })
    }

    /// Parse a reference string.
    #[staticmethod]
    fn parse(reference: &str) -> PyResult<Self> {
        let inner: llmns::Reference = reference.parse().map_err(|error: llmns::ParseError| {
            ParseError::new_err(format!("{error}: {reference:?}"))
        })?;
        Ok(PyReference { inner })
    }

    #[getter]
    fn tls(&self) -> bool {
        self.inner.tls
    }

    #[getter]
    fn transport(&self) -> &str {
        self.inner.transport_or_default()
    }

    #[getter]
    fn credential(&self) -> Option<&str> {
        self.inner.credential.as_deref()
    }

    #[getter]
    fn host(&self) -> &str {
        &self.inner.host
    }

    #[getter]
    fn port(&self) -> Option<u16> {
        self.inner.port
    }

    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    #[getter]
    fn pin(&self) -> Option<PyPin> {
        self.inner.pin.clone().map(|inner| PyPin { inner })
    }

    /// The hints as a dict; the first occurrence of a key applies.
    #[getter]
    fn hints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let hints = PyDict::new(py);
        for (key, value) in self.inner.hints() {
            if !hints.contains(key)? {
                hints.set_item(key, value)?;
            }
        }
        Ok(hints)
    }

    /// The reference with the host, model, and pin normalized per the
    /// equivalence rule; the credential and the hints are unchanged.
    fn normalized(&self) -> PyReference {
        PyReference {
            inner: self.inner.normalized(),
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Reference({:?})", self.inner.to_string())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyReference>() {
            Ok(o) => self.inner.is_equivalent(&o.inner),
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.identity().hash(&mut hasher);
        hasher.finish()
    }
}

/// Parse a reference string.
#[pyfunction]
fn parse(reference: &str) -> PyResult<PyReference> {
    PyReference::parse(reference)
}

#[pymodule]
fn _llmns(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("ParseError", py.get_type::<ParseError>())?;
    m.add_class::<PyPin>()?;
    m.add_class::<PyReference>()?;
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    Ok(())
}
