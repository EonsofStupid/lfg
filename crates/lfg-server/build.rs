use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn collect(root: &Path, directory: &Path, files: &mut Vec<(String, PathBuf)>) {
    println!("cargo:rerun-if-changed={}", directory.display());
    let entries = std::fs::read_dir(directory).expect("read UI asset directory");
    for entry in entries {
        let entry = entry.expect("read UI asset entry");
        let kind = entry.file_type().expect("read UI asset type");
        let path = entry.path();
        if kind.is_symlink() {
            panic!(
                "UI asset tree cannot contain symbolic links: {}",
                path.display()
            );
        } else if kind.is_dir() {
            collect(root, &path, files);
        } else if kind.is_file() {
            let relative = path.strip_prefix(root).expect("UI asset within root");
            let segments: Vec<_> = relative
                .iter()
                .map(|part| part.to_str().expect("UI asset names must be UTF-8"))
                .collect();
            let name = segments.join("/");
            assert!(!name.contains(['\n', '\r']), "invalid UI asset name");
            println!("cargo:rerun-if-changed={}", path.display());
            files.push((format!("/{name}"), path));
        }
    }
}

fn embed_ui() {
    let root = std::env::var_os("LFG_UI_DIST")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
                .join("../../ui/master-clyffy-web/dist")
        });
    let root = root
        .canonicalize()
        .expect("embedded UI assets not found; build the UI or set LFG_UI_DIST");
    assert!(
        root.join("index.html").is_file(),
        "embedded UI requires index.html"
    );
    let mut files = Vec::new();
    collect(&root, &root, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut source = String::from("pub static ASSETS: &[(&str, &[u8])] = &[\n");
    for (route, path) in files {
        source.push_str(&format!(
            "({route:?}, include_bytes!({:?}) as &[u8]),\n",
            path.to_string_lossy()
        ));
    }
    source.push_str("];\n");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("embedded_ui.rs");
    std::fs::write(out, source).expect("write embedded UI asset index");
}

fn embed_onnx() {
    let root = PathBuf::from(
        std::env::var_os("LFG_EMBEDDED_ONNX_DIR")
            .expect("embedded-onnx requires LFG_EMBEDDED_ONNX_DIR"),
    )
    .canonicalize()
    .expect("embedded ONNX directory not found");
    let expected = std::env::var("LFG_EMBEDDED_ONNX_SHA256")
        .expect("embedded-onnx requires LFG_EMBEDDED_ONNX_SHA256");
    assert!(
        expected.len() == 64
            && expected
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "embedded ONNX manifest digest must be lowercase SHA-256"
    );
    let names = [
        "manifest.json",
        "tokenizer.json",
        "labels.json",
        "prediction-config.json",
        "taxonomy.json",
        "model.onnx",
    ];
    let mut source = format!(
        "pub const MANIFEST_SHA256: &str = {expected:?};\npub static FILES: &[(&str, &[u8])] = &[\n"
    );
    for name in names {
        let path = root.join(name);
        let metadata =
            std::fs::symlink_metadata(&path).expect("embedded ONNX artifact file missing");
        assert!(
            metadata.file_type().is_file(),
            "embedded ONNX artifact must contain regular files"
        );
        println!("cargo:rerun-if-changed={}", path.display());
        if name == "manifest.json" {
            let bytes = std::fs::read(&path).expect("read embedded ONNX manifest");
            assert_eq!(
                format!("{:x}", Sha256::digest(bytes)),
                expected,
                "embedded ONNX manifest digest mismatch"
            );
        }
        source.push_str(&format!(
            "({name:?}, include_bytes!({:?}) as &[u8]),\n",
            path.to_string_lossy()
        ));
    }
    source.push_str("];\n");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("embedded_onnx.rs");
    std::fs::write(out, source).expect("write embedded ONNX artifact index");
}

fn embed_node() {
    let artifact = PathBuf::from(
        std::env::var_os("LFG_EMBEDDED_NODE_DIR")
            .expect("embedded-node requires LFG_EMBEDDED_NODE_DIR"),
    );
    let manifest_sha = std::env::var("LFG_EMBEDDED_NODE_MANIFEST_SHA256")
        .expect("embedded-node requires a manifest digest");
    let pack_sha = std::env::var("LFG_EMBEDDED_NODE_PACK_SHA256")
        .expect("embedded-node requires a pack digest");
    for hash in [&manifest_sha, &pack_sha] {
        assert!(
            hash.len() == 64
                && hash
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "embedded node digests must be lowercase SHA-256"
        );
    }
    let selected = |name: &str| {
        PathBuf::from(
            std::env::var_os(name).unwrap_or_else(|| panic!("embedded-node requires {name}")),
        )
    };
    let files = [
        ("manifest.json", artifact.join("manifest.json")),
        ("tokenizer.json", artifact.join("tokenizer.json")),
        ("model.onnx", artifact.join("model.onnx")),
        ("pack.json", selected("LFG_EMBEDDED_NODE_PACK")),
        ("snapshot.json", selected("LFG_EMBEDDED_NODE_SNAPSHOT")),
        ("runtime.json", selected("LFG_EMBEDDED_NODE_RUNTIME")),
        ("exemplar-config.json", selected("LFG_EMBEDDED_NODE_CONFIG")),
    ];
    let mut source = format!(
        "pub const MANIFEST_SHA256: &str = {manifest_sha:?};\npub const PACK_SHA256: &str = {pack_sha:?};\npub static FILES: &[(&str, &[u8])] = &[\n"
    );
    for (name, path) in files {
        let metadata =
            std::fs::symlink_metadata(&path).expect("embedded node artifact file missing");
        assert!(
            metadata.file_type().is_file(),
            "embedded node inputs must be regular files"
        );
        let path = path.canonicalize().expect("resolve embedded node input");
        println!("cargo:rerun-if-changed={}", path.display());
        if name == "manifest.json" || name == "pack.json" {
            let bytes = std::fs::read(&path).expect("read pinned embedded node input");
            let expected = if name == "manifest.json" {
                &manifest_sha
            } else {
                &pack_sha
            };
            assert_eq!(
                &format!("{:x}", Sha256::digest(bytes)),
                expected,
                "embedded node input digest mismatch"
            );
        }
        source.push_str(&format!(
            "({name:?}, include_bytes!({:?}) as &[u8]),\n",
            path.to_string_lossy()
        ));
    }
    source.push_str("];\n");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("embedded_node.rs");
    std::fs::write(out, source).expect("write embedded node index");
}

fn embed_owned_byte() {
    let root = PathBuf::from(
        std::env::var_os("LFG_EMBEDDED_OWNED_BYTE_DIR")
            .expect("embedded-owned-byte requires LFG_EMBEDDED_OWNED_BYTE_DIR"),
    )
    .canonicalize()
    .expect("resolve owned byte artifact directory");
    let expected = std::env::var("LFG_EMBEDDED_OWNED_BYTE_MANIFEST_SHA256")
        .expect("embedded-owned-byte requires a manifest digest");
    assert!(
        expected.len() == 64
            && expected
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "owned byte manifest digest must be lowercase SHA-256"
    );
    let mut source = format!("pub const MANIFEST_SHA256: &str = {expected:?};\n");
    for (constant, name) in [("MANIFEST", "manifest.json"), ("MODEL", "model.onnx")] {
        let path = root.join(name);
        assert!(
            std::fs::symlink_metadata(&path)
                .expect("owned byte artifact file missing")
                .file_type()
                .is_file(),
            "owned byte artifact must contain regular files"
        );
        if name == "manifest.json" {
            assert_eq!(
                format!(
                    "{:x}",
                    Sha256::digest(std::fs::read(&path).expect("read manifest"))
                ),
                expected,
                "owned byte manifest digest mismatch"
            );
        }
        println!("cargo:rerun-if-changed={}", path.display());
        source.push_str(&format!(
            "pub static {constant}: &[u8] = include_bytes!({:?});\n",
            path.to_string_lossy()
        ));
    }
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("embedded_owned_byte.rs");
    std::fs::write(out, source).expect("write owned byte artifact index");
}

fn digest_file(path: &Path) -> String {
    use std::io::Read;
    let mut file = std::fs::File::open(path).expect("open cached ONNX graph for digest");
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).expect("read cached ONNX graph");
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    format!("{:x}", hasher.finalize())
}

fn embed_owned_cached_byte() {
    const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
    let root = PathBuf::from(
        std::env::var_os("LFG_EMBEDDED_OWNED_CACHED_DIR")
            .expect("embedded-owned-cached-byte requires LFG_EMBEDDED_OWNED_CACHED_DIR"),
    )
    .canonicalize()
    .expect("resolve embedded cached artifact directory");
    let expected_manifest_sha = std::env::var("LFG_EMBEDDED_OWNED_CACHED_MANIFEST_SHA256")
        .expect("embedded-owned-cached-byte requires a pinned manifest digest");
    let max_total_graph_bytes = std::env::var("LFG_EMBEDDED_OWNED_CACHED_MAX_TOTAL_BYTES")
        .expect("embedded-owned-cached-byte requires a total graph byte limit")
        .parse::<u64>()
        .expect("embedded cached graph byte limit must be a positive integer");
    assert!(
        max_total_graph_bytes > 0,
        "embedded cached graph byte limit must be positive"
    );
    assert!(
        expected_manifest_sha.len() == 64
            && expected_manifest_sha
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "embedded cached manifest digest must be lowercase SHA-256"
    );
    let manifest_path = root.join("manifest.json");
    let manifest_meta =
        std::fs::symlink_metadata(&manifest_path).expect("embedded cached manifest missing");
    assert!(
        manifest_meta.file_type().is_file() && manifest_meta.len() <= MAX_MANIFEST_BYTES,
        "embedded cached manifest must be a bounded regular file"
    );
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    let manifest_raw = std::fs::read(&manifest_path).expect("read embedded cached manifest");
    assert_eq!(
        format!("{:x}", Sha256::digest(&manifest_raw)),
        expected_manifest_sha,
        "embedded cached manifest pin mismatch"
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_raw).expect("parse embedded cached manifest");
    let mut total = 0_u64;
    let mut source = format!("pub const MANIFEST_SHA256: &str = {expected_manifest_sha:?};\n");
    for (constant, name, digest_key) in [
        ("PREFILL", "prefill.onnx", "prefill_onnx_sha256"),
        ("DECODE", "decode.onnx", "decode_onnx_sha256"),
    ] {
        let path = root.join(name);
        let metadata = std::fs::symlink_metadata(&path).expect("embedded cached graph missing");
        assert!(
            metadata.file_type().is_file(),
            "embedded cached graph must be a regular file"
        );
        total = total
            .checked_add(metadata.len())
            .expect("embedded cached graph size overflow");
        assert!(
            total <= max_total_graph_bytes,
            "embedded cached graphs exceed configured size"
        );
        println!("cargo:rerun-if-changed={}", path.display());
        let path = path.canonicalize().expect("resolve embedded cached graph");
        let actual_sha = digest_file(&path);
        let expected_sha = manifest[digest_key]
            .as_str()
            .expect("cached manifest graph digest missing");
        assert_eq!(
            actual_sha, expected_sha,
            "embedded cached graph digest differs from manifest"
        );
        source.push_str(&format!(
            "pub static {constant}: &[u8] = include_bytes!({:?});\n",
            path.to_string_lossy()
        ));
    }
    source.push_str(&format!(
        "pub static MANIFEST: &[u8] = include_bytes!({:?});\n",
        manifest_path
            .canonicalize()
            .expect("resolve embedded cached manifest")
            .to_string_lossy()
    ));
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("embedded_owned_cached.rs");
    std::fs::write(out, source).expect("write embedded cached artifact index");
}

fn main() {
    println!("cargo:rerun-if-env-changed=LFG_UI_DIST");
    println!("cargo:rerun-if-env-changed=LFG_EMBEDDED_ONNX_DIR");
    println!("cargo:rerun-if-env-changed=LFG_EMBEDDED_ONNX_SHA256");
    println!("cargo:rerun-if-env-changed=LFG_EMBEDDED_OWNED_BYTE_DIR");
    println!("cargo:rerun-if-env-changed=LFG_EMBEDDED_OWNED_BYTE_MANIFEST_SHA256");
    println!("cargo:rerun-if-env-changed=LFG_EMBEDDED_OWNED_CACHED_DIR");
    println!("cargo:rerun-if-env-changed=LFG_EMBEDDED_OWNED_CACHED_MANIFEST_SHA256");
    println!("cargo:rerun-if-env-changed=LFG_EMBEDDED_OWNED_CACHED_MAX_TOTAL_BYTES");
    for name in [
        "LFG_EMBEDDED_NODE_DIR",
        "LFG_EMBEDDED_NODE_MANIFEST_SHA256",
        "LFG_EMBEDDED_NODE_PACK",
        "LFG_EMBEDDED_NODE_PACK_SHA256",
        "LFG_EMBEDDED_NODE_SNAPSHOT",
        "LFG_EMBEDDED_NODE_RUNTIME",
        "LFG_EMBEDDED_NODE_CONFIG",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    if std::env::var_os("CARGO_FEATURE_EMBEDDED_UI").is_some() {
        embed_ui();
    }
    if std::env::var_os("CARGO_FEATURE_EMBEDDED_ONNX").is_some() {
        embed_onnx();
    }
    if std::env::var_os("CARGO_FEATURE_EMBEDDED_NODE").is_some() {
        embed_node();
    }
    if std::env::var_os("CARGO_FEATURE_EMBEDDED_OWNED_BYTE").is_some() {
        embed_owned_byte();
    }
    if std::env::var_os("CARGO_FEATURE_EMBEDDED_OWNED_CACHED_BYTE").is_some() {
        embed_owned_cached_byte();
    }
}
