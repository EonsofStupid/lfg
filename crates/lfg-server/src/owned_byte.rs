//! Experimental offline byte-model probe. This does not assess or route prompts.
use std::{
    borrow::Cow,
    collections::HashSet,
    fs::File,
    io::{self, BufRead, Read, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tract::prelude::*;

#[cfg(feature = "embedded-owned-byte")]
mod embedded {
    include!(concat!(env!("OUT_DIR"), "/embedded_owned_byte.rs"));
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
struct Config {
    schema_version: String,
    max_manifest_bytes: u64,
    max_model_bytes: u64,
    max_context: usize,
    max_line_bytes: usize,
    max_examples: usize,
    max_generation_bytes: Option<usize>,
}

#[derive(Deserialize)]
struct Manifest {
    schema_version: String,
    promotion_eligible: bool,
    model_sha256: String,
    context: usize,
    vocabulary: usize,
    input: String,
    output: String,
}

#[derive(Deserialize)]
struct Request {
    id: String,
    text: String,
}

#[derive(Serialize)]
struct Response<'a> {
    schema_version: &'static str,
    id: &'a str,
    input_bytes: usize,
    next_byte: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    logits: Option<&'a [f32]>,
}

#[derive(Serialize)]
struct Generation<'a> {
    schema_version: &'static str,
    id: &'a str,
    input_bytes: usize,
    generated_bytes: &'a [u8],
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    stop_reason: &'static str,
}

enum Mode<'a> {
    Probe {
        include_logits: bool,
    },
    Generate {
        limit: usize,
        stop: Option<&'a [u8]>,
    },
}

fn digest(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

fn read_pinned_config(path: &Path, maximum_bytes: u64, expected_sha256: &str) -> Result<Vec<u8>> {
    if maximum_bytes == 0
        || expected_sha256.len() != 64
        || !expected_sha256.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("invalid pinned runtime configuration limit or digest".into());
    }
    let file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err("runtime configuration must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(
        maximum_bytes
            .checked_add(1)
            .ok_or("runtime configuration limit overflow")?,
    )
    .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum_bytes {
        return Err("runtime configuration exceeds pinned byte limit".into());
    }
    if digest(&bytes) != expected_sha256.to_ascii_lowercase() {
        return Err("runtime configuration digest mismatch".into());
    }
    Ok(bytes)
}

fn greedy(logits: &[f32]) -> Result<u8> {
    if logits.len() != 256 || logits.iter().any(|value| !value.is_finite()) {
        return Err("invalid owned byte logits".into());
    }
    // Strictly greater preserves the first index when logits tie, matching torch.argmax.
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    Ok(best as u8)
}

fn generate_bytes<F>(
    input: &[u8],
    context_limit: usize,
    limit: usize,
    stop: Option<&[u8]>,
    mut predict: F,
) -> Result<(Vec<u8>, &'static str)>
where
    F: FnMut(&[u8]) -> Result<u8>,
{
    let mut context = input.to_vec();
    let mut generated = Vec::new();
    while generated.len() < limit {
        if context.len() > context_limit {
            return Ok((generated, "context_limit"));
        }
        let next = predict(&context)?;
        context.push(next);
        generated.push(next);
        if stop.is_some_and(|value| generated.ends_with(value)) {
            return Ok((generated, "stop_sequence"));
        }
    }
    Ok((generated, "byte_limit"))
}

fn bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    if !std::fs::symlink_metadata(path)?.file_type().is_file() {
        return Err("artifact must be a regular file".into());
    }
    let mut raw = Vec::new();
    File::open(path)?
        .take(limit.saturating_add(1))
        .read_to_end(&mut raw)?;
    if raw.len() as u64 > limit {
        return Err("artifact exceeds configured byte limit".into());
    }
    Ok(raw)
}

enum Artifact<'a> {
    Directory(&'a Path),
    #[cfg(feature = "embedded-owned-byte")]
    Embedded,
}

impl Artifact<'_> {
    fn read(&self, name: &str, limit: u64) -> Result<Cow<'_, [u8]>> {
        match self {
            Self::Directory(path) => Ok(Cow::Owned(bounded(&path.join(name), limit)?)),
            #[cfg(feature = "embedded-owned-byte")]
            Self::Embedded => {
                let data = match name {
                    "manifest.json" => embedded::MANIFEST,
                    "model.onnx" => embedded::MODEL,
                    _ => return Err("unknown artifact file".into()),
                };
                if data.len() as u64 > limit {
                    return Err("embedded artifact exceeds configured byte limit".into());
                }
                Ok(Cow::Borrowed(data))
            }
        }
    }
}

fn execute(
    artifact: Artifact<'_>,
    manifest_sha: &str,
    config_path: Option<&Path>,
    runtime_config_sha256: Option<&str>,
    runtime_config_max_bytes: Option<u64>,
    mode: Mode<'_>,
) -> Result<()> {
    let raw_config: Cow<'_, [u8]> = match config_path {
        Some(path) => match (runtime_config_sha256, runtime_config_max_bytes) {
            (Some(sha), Some(max)) => Cow::Owned(read_pinned_config(path, max, sha)?),
            (None, None) => Cow::Owned(std::fs::read(path)?),
            _ => {
                return Err(
                    "runtime config digest and byte limit must be provided together".into(),
                );
            }
        },
        None if runtime_config_sha256.is_some() || runtime_config_max_bytes.is_some() => {
            return Err("runtime config pin requires a config file".into());
        }
        None => Cow::Borrowed(include_bytes!("../config/owned-byte-runtime.example.json")),
    };
    let config: Config = serde_json::from_slice(&raw_config)?;
    if config.schema_version != "lfg.owned-byte-runtime.v1"
        || config.max_manifest_bytes == 0
        || config.max_model_bytes == 0
        || config.max_context == 0
        || config.max_line_bytes == 0
        || config.max_examples == 0
    {
        return Err("invalid owned byte runtime configuration".into());
    }
    if let Mode::Generate { limit, stop } = &mode {
        if *limit == 0
            || config
                .max_generation_bytes
                .is_none_or(|maximum| *limit > maximum)
            || stop.is_some_and(|value| value.is_empty() || value.len() > *limit)
        {
            return Err("invalid or unconfigured generation limit or stop sequence".into());
        }
    }
    let manifest_raw = artifact.read("manifest.json", config.max_manifest_bytes)?;
    if digest(&manifest_raw) != manifest_sha {
        return Err("owned byte manifest digest mismatch".into());
    }
    let manifest: Manifest = serde_json::from_slice(&manifest_raw)?;
    if manifest.schema_version != "lfg.owned-byte-onnx.v1"
        || manifest.promotion_eligible
        || manifest.context == 0
        || manifest.context > config.max_context
        || manifest.vocabulary != 256
        || manifest.input != "tokens"
        || manifest.output != "logits"
    {
        return Err("unsupported owned byte manifest".into());
    }
    let bytes = artifact.read("model.onnx", config.max_model_bytes)?;
    if digest(&bytes) != manifest.model_sha256 {
        return Err("owned byte model digest mismatch".into());
    }
    let mut model = tract::onnx()?.load_buffer(&bytes)?;
    // tract exposes the producing node name, not necessarily the ONNX output label.
    // Identity comes from the model digest; output count and shape are checked.
    if model.input_count()? != 1
        || model.output_count()? != 1
        || model.input_name(0)? != manifest.input
    {
        return Err("owned byte graph inputs or outputs differ from manifest".into());
    }
    model.set_input_fact(0, "1,S,i64")?;
    let runnable = model.into_model()?.into_runnable()?;
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = stdout.lock();
    let mut line = Vec::new();
    let mut ids = HashSet::new();
    loop {
        line.clear();
        let count = reader
            .by_ref()
            .take((config.max_line_bytes as u64).saturating_add(1))
            .read_until(b'\n', &mut line)?;
        if count == 0 {
            break;
        }
        if count > config.max_line_bytes {
            return Err("input line exceeds configured limit".into());
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let request: Request = serde_json::from_slice(&line)?;
        if request.id.trim().is_empty()
            || ids.len() >= config.max_examples
            || !ids.insert(request.id.clone())
        {
            return Err("empty or duplicate ID, or example limit exceeded".into());
        }
        let input = request.text.as_bytes();
        if input.is_empty() || input.len() > manifest.context {
            return Err("input exceeds model context or is empty".into());
        }
        let logits_for = |bytes: &[u8]| -> Result<Vec<f32>> {
            let ids: Vec<i64> = bytes.iter().map(|&byte| i64::from(byte)).collect();
            let outputs = runnable.run([Tensor::from_slice(&[1, bytes.len()], &ids)?])?;
            if outputs.len() != 1 || outputs[0].shape()? != [1, bytes.len(), 256] {
                return Err("invalid owned byte output shape".into());
            }
            let all = outputs[0].as_slice::<f32>()?;
            let logits = &all[all.len() - 256..];
            if logits.iter().any(|value| !value.is_finite()) {
                return Err("non-finite logits".into());
            }
            Ok(logits.to_vec())
        };
        match &mode {
            Mode::Probe { include_logits } => {
                let logits = logits_for(input)?;
                serde_json::to_writer(
                    &mut writer,
                    &Response {
                        schema_version: "lfg.owned-byte-probe.v1",
                        id: &request.id,
                        input_bytes: input.len(),
                        next_byte: greedy(&logits)?,
                        logits: include_logits.then_some(logits.as_slice()),
                    },
                )?;
            }
            Mode::Generate { limit, stop } => {
                let (generated, reason) =
                    generate_bytes(input, manifest.context, *limit, *stop, |bytes| {
                        greedy(&logits_for(bytes)?)
                    })?;
                serde_json::to_writer(
                    &mut writer,
                    &Generation {
                        schema_version: "lfg.owned-byte-generation.v1",
                        id: &request.id,
                        input_bytes: input.len(),
                        generated_bytes: &generated,
                        text: std::str::from_utf8(&generated).ok(),
                        stop_reason: reason,
                    },
                )?;
            }
        }
        writeln!(writer)?;
    }
    if ids.is_empty() {
        return Err("no examples supplied".into());
    }
    Ok(())
}

pub fn directory(
    root: &Path,
    manifest_sha: &str,
    config: &Path,
    include_logits: bool,
) -> Result<()> {
    execute(
        Artifact::Directory(root),
        manifest_sha,
        Some(config),
        None,
        None,
        Mode::Probe { include_logits },
    )
}

#[cfg(feature = "embedded-owned-byte")]
pub fn embedded(config: Option<&Path>, include_logits: bool) -> Result<()> {
    execute(
        Artifact::Embedded,
        embedded::MANIFEST_SHA256,
        config,
        None,
        None,
        Mode::Probe { include_logits },
    )
}

pub fn generate_directory(
    root: &Path,
    manifest_sha: &str,
    config: &Path,
    limit: usize,
    stop: Option<&str>,
    runtime_config_sha256: Option<&str>,
    runtime_config_max_bytes: Option<u64>,
) -> Result<()> {
    execute(
        Artifact::Directory(root),
        manifest_sha,
        Some(config),
        runtime_config_sha256,
        runtime_config_max_bytes,
        Mode::Generate {
            limit,
            stop: stop.map(str::as_bytes),
        },
    )
}

#[cfg(feature = "embedded-owned-byte")]
pub fn generate_embedded(config: Option<&Path>, limit: usize, stop: Option<&str>) -> Result<()> {
    execute(
        Artifact::Embedded,
        embedded::MANIFEST_SHA256,
        config,
        None,
        None,
        Mode::Generate {
            limit,
            stop: stop.map(str::as_bytes),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_uses_first_tied_byte_and_rejects_invalid_logits() {
        let mut logits = vec![0.0; 256];
        logits[3] = 1.0;
        logits[7] = 1.0;
        assert_eq!(greedy(&logits).unwrap(), 3);
        logits[0] = f32::NAN;
        assert!(greedy(&logits).is_err());
        assert!(greedy(&[0.0]).is_err());
    }

    #[test]
    fn generation_stops_at_sequence_byte_and_context_limits() {
        let mut next = [b'A', b'B', b'C'].into_iter();
        let (bytes, reason) =
            generate_bytes(b"x", 8, 3, Some(b"AB"), |_| Ok(next.next().unwrap())).unwrap();
        assert_eq!(
            (bytes.as_slice(), reason),
            (b"AB".as_slice(), "stop_sequence")
        );
        let (bytes, reason) = generate_bytes(b"x", 8, 2, None, |_| Ok(b'A')).unwrap();
        assert_eq!((bytes.as_slice(), reason), (b"AA".as_slice(), "byte_limit"));
        let (bytes, reason) = generate_bytes(b"xy", 2, 3, None, |_| Ok(b'A')).unwrap();
        assert_eq!(
            (bytes.as_slice(), reason),
            (b"A".as_slice(), "context_limit")
        );
    }

    #[test]
    fn pinned_runtime_config_rejects_changed_and_oversized_files_before_parse() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("runtime.json");
        let original = br#"{"schema_version":"lfg.owned-byte-runtime.v1"}"#;
        std::fs::write(&path, original).unwrap();
        let pinned = digest(original);
        assert_eq!(
            read_pinned_config(&path, original.len() as u64, &pinned).unwrap(),
            original
        );
        std::fs::write(&path, br#"{"schema_version":"changed"}"#).unwrap();
        assert!(read_pinned_config(&path, 1024, &pinned).is_err());
        std::fs::write(&path, vec![b'x'; 32]).unwrap();
        assert!(read_pinned_config(&path, 8, &pinned).is_err());
    }
}
