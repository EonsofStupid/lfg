//! Shared loopback, Host, Origin and high-entropy token checks for private local APIs.
use std::net::SocketAddr;

use axum::http::{HeaderMap, header};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LocalAccessError {
    NonLoopback,
    InvalidHost,
    InvalidOrigin,
    TokenUnavailable,
    InvalidToken,
}

pub(crate) fn valid_token(token: &str) -> bool {
    token.len() >= 32
        && token.len() <= 4096
        && !token
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || byte.is_ascii_control())
        && token
            .bytes()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            >= 12
}

pub(crate) fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        difference |= (left.get(index).copied().unwrap_or(0)
            ^ right.get(index).copied().unwrap_or(0)) as usize;
    }
    difference == 0
}

pub(crate) fn valid_origin(value: &str) -> bool {
    let Some(authority) = value.strip_prefix("http://") else {
        return false;
    };
    if authority.contains('/')
        || authority.contains('@')
        || authority.contains('#')
        || authority.contains('?')
    {
        return false;
    }
    let host = authority
        .rsplit_once(':')
        .map(|(host, _)| host)
        .unwrap_or(authority);
    matches!(host, "localhost" | "127.0.0.1" | "[::1]")
}

pub(crate) fn valid_authority(value: &str) -> bool {
    if value.contains(['/', '@', '#', '?']) {
        return false;
    }
    let host = value
        .rsplit_once(':')
        .map(|(host, _)| host)
        .unwrap_or(value);
    matches!(host, "localhost" | "127.0.0.1" | "[::1]")
}

pub(crate) fn authorize(
    listener: SocketAddr,
    peer: SocketAddr,
    headers: &HeaderMap,
    token: Option<&str>,
    allowed_hosts: &[String],
    allowed_origins: &[String],
) -> Result<(), LocalAccessError> {
    if !listener.ip().is_loopback() || !peer.ip().is_loopback() {
        return Err(LocalAccessError::NonLoopback);
    }
    let authority = listener.to_string();
    let host = headers
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    if host != authority && !allowed_hosts.iter().any(|allowed| allowed == host) {
        return Err(LocalAccessError::InvalidHost);
    }
    if let Some(origin) = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
        let expected = format!("http://{authority}");
        if origin != expected && !allowed_origins.iter().any(|allowed| allowed == origin) {
            return Err(LocalAccessError::InvalidOrigin);
        }
    }
    let Some(token) = token else {
        return Err(LocalAccessError::TokenUnavailable);
    };
    let supplied = headers
        .get("x-lfg-session-token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !constant_time_eq(supplied.as_bytes(), token.as_bytes()) {
        return Err(LocalAccessError::InvalidToken);
    }
    Ok(())
}
