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

fn main() {
    println!("cargo:rerun-if-env-changed=LFG_UI_DIST");
    if std::env::var_os("CARGO_FEATURE_EMBEDDED_UI").is_none() {
        return;
    }
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
