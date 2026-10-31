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

fn main() {
    println!("cargo:rerun-if-env-changed=LFG_UI_DIST");
    println!("cargo:rerun-if-env-changed=LFG_EMBEDDED_ONNX_DIR");
    println!("cargo:rerun-if-env-changed=LFG_EMBEDDED_ONNX_SHA256");
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
}
