use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    boundary_questions::{Attunement, Catalog},
    intake::{self, Template},
};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Config {
    schema_version: String,
    revision: String,
    source_app: String,
    selections: Vec<Value>,
    limits: Limits,
    sources: Vec<Source>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Limits {
    max_entries: usize,
    max_depth: usize,
    max_file_bytes: u64,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum Source {
    Inventory {
        binding: String,
        path: String,
        recursive: bool,
        extensions: Vec<String>,
    },
    Json {
        binding: String,
        path: String,
        pointer: String,
    },
    Text {
        binding: String,
        path: String,
    },
}

impl Source {
    fn binding(&self) -> &str {
        match self {
            Self::Inventory { binding, .. }
            | Self::Json { binding, .. }
            | Self::Text { binding, .. } => binding,
        }
    }
    fn path(&self) -> &str {
        match self {
            Self::Inventory { path, .. } | Self::Json { path, .. } | Self::Text { path, .. } => {
                path
            }
        }
    }
}

pub(crate) fn run(
    root: &Path,
    text: &str,
    catalog: &Catalog,
    intake: &Template,
) -> Result<Value, String> {
    let config: Config = serde_json::from_str(text).map_err(|e| e.to_string())?;
    if config.schema_version != "lfg.project-attunement.v1"
        || config.revision.trim().is_empty()
        || config.source_app.trim().is_empty()
        || config.sources.is_empty()
        || config.limits.max_entries == 0
        || config.limits.max_file_bytes == 0
    {
        return Err("invalid project attunement configuration".into());
    }
    let mut bindings = BTreeSet::new();
    for source in &config.sources {
        if !bindings.insert(source.binding()) || !catalog.has_binding(source.binding()) {
            return Err(format!(
                "duplicate or unknown binding: {}",
                source.binding()
            ));
        }
        validate_relative(source.path())?;
        match source {
            Source::Inventory { extensions, .. }
                if extensions
                    .iter()
                    .any(|s| s.is_empty() || s.contains(['/', '\\', '.'])) =>
            {
                return Err("invalid inventory extension".into());
            }
            Source::Json { pointer, .. } if !intake::valid_pointer(pointer) => {
                return Err("invalid JSON pointer".into());
            }
            _ => {}
        }
    }
    let root = fs::canonicalize(root).map_err(|e| format!("cannot open project root: {e}"))?;
    if !root.is_dir() {
        return Err("project root is not a directory".into());
    }
    let mut artifacts = BTreeMap::<String, Value>::new();
    let mut provenance = Vec::new();
    let mut entries = 0usize;
    for source in &config.sources {
        let Some(path) = checked_path(&root, source.path())? else {
            provenance
                .push(json!({"binding":source.binding(),"path":source.path(),"status":"missing"}));
            continue;
        };
        match source {
            Source::Inventory {
                binding,
                recursive,
                extensions,
                ..
            } => {
                if !path.is_dir() {
                    return Err(format!(
                        "inventory path is not a directory: {}",
                        source.path()
                    ));
                }
                let mut files = Vec::new();
                Inventory {
                    root: &root,
                    extensions,
                    limits: &config.limits,
                    entries: &mut entries,
                    files: &mut files,
                }
                .visit(&path, 0, *recursive)?;
                files.sort();
                let digest = lfg_core::digest_value(&json!(files));
                provenance.push(json!({"binding":binding,"path":source.path(),"kind":"inventory","status":"collected","recursive":recursive,"extensions":extensions,"fileCount":files.len(),"pathsDigest":digest}));
                artifacts.insert(binding.clone(), json!(files));
            }
            Source::Json {
                binding, pointer, ..
            } => {
                if !path.is_file() {
                    return Err(format!("JSON source is not a file: {}", source.path()));
                }
                let mut bytes = Vec::new();
                fs::File::open(&path)
                    .map_err(|e| e.to_string())?
                    .take(config.limits.max_file_bytes.saturating_add(1))
                    .read_to_end(&mut bytes)
                    .map_err(|e| e.to_string())?;
                if bytes.len() as u64 > config.limits.max_file_bytes {
                    return Err("JSON source exceeds maxFileBytes".into());
                }
                let document: Value = serde_json::from_slice(&bytes)
                    .map_err(|e| format!("invalid JSON source {}: {e}", source.path()))?;
                let value = document.pointer(pointer);
                provenance.push(json!({"binding":binding,"path":source.path(),"kind":"json","pointer":pointer,"status":if value.is_some() {"collected"} else {"missing"},"sha256":format!("{:x}",Sha256::digest(&bytes))}));
                if let Some(value) = value {
                    artifacts.insert(binding.clone(), value.clone());
                }
            }
            Source::Text { binding, .. } => {
                if !path.is_file() {
                    return Err(format!("text source is not a file: {}", source.path()));
                }
                let mut bytes = Vec::new();
                fs::File::open(&path)
                    .map_err(|e| e.to_string())?
                    .take(config.limits.max_file_bytes.saturating_add(1))
                    .read_to_end(&mut bytes)
                    .map_err(|e| e.to_string())?;
                if bytes.len() as u64 > config.limits.max_file_bytes {
                    return Err("text source exceeds maxFileBytes".into());
                }
                let content = std::str::from_utf8(&bytes)
                    .map_err(|_| format!("text source is not UTF-8: {}", source.path()))?;
                provenance.push(json!({"binding":binding,"path":source.path(),"kind":"text","status":"collected","sha256":format!("{:x}",Sha256::digest(&bytes))}));
                artifacts.insert(binding.clone(), Value::String(content.to_owned()));
            }
        }
    }
    let request: Attunement = serde_json::from_value(json!({"schemaVersion":"lfg.boundary-attunement.v1","sourceApp":config.source_app,"selections":config.selections,"artifacts":artifacts})).map_err(|e| e.to_string())?;
    let preview = catalog.attune(request, intake)?;
    Ok(
        json!({"schemaVersion":"lfg.project-attunement-preview.v1","configRevision":config.revision,"configSha256":format!("{:x}",Sha256::digest(text.as_bytes())),"sources":provenance,"preview":preview}),
    )
}

fn validate_relative(path: &str) -> Result<(), String> {
    if path.is_empty()
        || !Path::new(path)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
    {
        return Err(format!(
            "source path must be a relative path without dot components: {path}"
        ));
    }
    Ok(())
}

fn checked_path(root: &Path, relative: &str) -> Result<Option<PathBuf>, String> {
    let mut path = root.to_path_buf();
    for component in Path::new(relative).components() {
        path.push(component.as_os_str());
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!("symlink source is not permitted: {relative}"));
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(format!("cannot inspect {relative}: {e}")),
        }
    }
    let canonical = fs::canonicalize(&path).map_err(|e| e.to_string())?;
    if !canonical.starts_with(root) {
        return Err("source escapes project root".into());
    }
    Ok(Some(canonical))
}

struct Inventory<'a> {
    root: &'a Path,
    extensions: &'a [String],
    limits: &'a Limits,
    entries: &'a mut usize,
    files: &'a mut Vec<String>,
}

impl Inventory<'_> {
    fn visit(&mut self, dir: &Path, depth: usize, recursive: bool) -> Result<(), String> {
        for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            *self.entries = self.entries.checked_add(1).ok_or("entry count overflow")?;
            if *self.entries > self.limits.max_entries {
                return Err("inventory exceeds maxEntries".into());
            }
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            if kind.is_symlink() {
                return Err("inventory contains a symlink".into());
            }
            if kind.is_dir() && recursive {
                if depth >= self.limits.max_depth {
                    return Err("inventory exceeds maxDepth".into());
                }
                self.visit(&entry.path(), depth + 1, true)?;
            } else if kind.is_file() {
                let path = entry.path();
                if self.extensions.is_empty()
                    || path
                        .extension()
                        .and_then(|v| v.to_str())
                        .is_some_and(|v| self.extensions.iter().any(|e| e == v))
                {
                    let rel = path.strip_prefix(self.root).map_err(|e| e.to_string())?;
                    let parts = rel
                        .components()
                        .map(|c| c.as_os_str().to_str().ok_or("non-UTF-8 inventory path"))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.files.push(parts.join("/"));
                }
            } else if !kind.is_dir() {
                return Err("inventory contains an unsupported file type".into());
            }
        }
        Ok(())
    }
}
