use axum::{
    body::Body,
    http::{Method, StatusCode, Uri, header},
    response::Response,
};

include!(concat!(env!("OUT_DIR"), "/embedded_ui.rs"));

fn mime(path: &str) -> &'static str {
    match path.rsplit_once('.').map(|(_, extension)| extension) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") | Some("mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}

pub async fn serve(method: Method, uri: Uri) -> Response {
    if method != Method::GET && method != Method::HEAD {
        return Response::builder()
            .status(StatusCode::METHOD_NOT_ALLOWED)
            .body(Body::empty())
            .unwrap();
    }
    let path = uri.path();
    if path == "/api" || path.starts_with("/api/") {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::empty())
            .unwrap();
    }
    let asset = ASSETS
        .binary_search_by_key(&path, |(name, _)| *name)
        .ok()
        .map(|index| ASSETS[index]);
    let asset = asset.or_else(|| {
        if path
            .rsplit('/')
            .next()
            .is_some_and(|part| !part.contains('.'))
        {
            ASSETS
                .binary_search_by_key(&"/index.html", |(name, _)| *name)
                .ok()
                .map(|index| ASSETS[index])
        } else {
            None
        }
    });
    match asset {
        Some((name, bytes)) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime(name))
            .header(header::CONTENT_LENGTH, bytes.len())
            .header(header::X_CONTENT_TYPE_OPTIONS, "nosniff")
            .body(if method == Method::HEAD {
                Body::empty()
            } else {
                Body::from(bytes)
            })
            .unwrap(),
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::empty())
            .unwrap(),
    }
}
