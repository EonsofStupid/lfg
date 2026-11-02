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

fn digest(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
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
    include_logits: bool,
) -> Result<()> {
    let raw_config: Cow<'_, [u8]> = match config_path {
        Some(path) => Cow::Owned(std::fs::read(path)?),
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
        let token_ids: Vec<i64> = input.iter().map(|&byte| i64::from(byte)).collect();
        let outputs = runnable.run([Tensor::from_slice(&[1, input.len()], &token_ids)?])?;
        if outputs.len() != 1 || outputs[0].shape()? != [1, input.len(), 256] {
            return Err("invalid owned byte output shape".into());
        }
        let all_logits = outputs[0].as_slice::<f32>()?;
        let logits = &all_logits[all_logits.len() - 256..];
        if logits.iter().any(|v| !v.is_finite()) {
            return Err("non-finite logits".into());
        }
        let index = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .ok_or("empty logits")?
            .0;
        serde_json::to_writer(
            &mut writer,
            &Response {
                schema_version: "lfg.owned-byte-probe.v1",
                id: &request.id,
                input_bytes: input.len(),
                next_byte: index as u8,
                logits: include_logits.then_some(logits),
            },
        )?;
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
        include_logits,
    )
}

#[cfg(feature = "embedded-owned-byte")]
pub fn embedded(config: Option<&Path>, include_logits: bool) -> Result<()> {
    execute(
        Artifact::Embedded,
        embedded::MANIFEST_SHA256,
        config,
        include_logits,
    )
}
