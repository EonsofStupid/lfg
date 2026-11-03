//! Private, local assessment feedback journal.
//!
//! The API is mounted only after the process has bound a loopback listener. Every
//! request also has to arrive from a loopback peer and use the listener authority.
#[cfg(not(windows))]
use std::path::Path;
use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use axum::{
    Json, Router,
    extract::{ConnectInfo, Path as AxumPath, Query, State},
    http::{HeaderMap, StatusCode, header},
    routing::{get, post},
};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde_json::{Value, json};

const DEFAULT_MAX_RECORD_BYTES: usize = 256 * 1024;
const DEFAULT_MAX_JOURNAL_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_PAGE_SIZE: usize = 50;
const DEFAULT_MAX_PAGE_SIZE: usize = 100;
const HARD_MAX_PAGE_SIZE: usize = 10_000;

#[derive(Clone)]
pub struct FeedbackApi {
    store: Option<FeedbackStore>,
    listener: SocketAddr,
    token: Option<Arc<String>>,
    origins: Arc<Vec<String>>,
    hosts: Arc<Vec<String>>,
    unavailable_reason: Option<String>,
}

#[derive(Clone)]
struct FeedbackStore {
    db: Arc<Mutex<Connection>>,
    #[cfg(windows)]
    _guard: Arc<super::feedback_windows::PrivateDirectory>,
    #[cfg(not(windows))]
    _guard: (),
    max_record_bytes: usize,
    max_journal_bytes: u64,
    max_page_size: usize,
}

impl FeedbackApi {
    pub fn for_listener(listener: SocketAddr) -> Self {
        let origins = std::env::var("LFG_FEEDBACK_ALLOWED_ORIGINS")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .filter(|value| local_http_origin(value))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let hosts = std::env::var("LFG_FEEDBACK_ALLOWED_HOSTS")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|value| local_authority(value))
            .map(str::to_owned)
            .take(16)
            .collect::<Vec<_>>();
        if !listener.ip().is_loopback() {
            return Self {
                store: None,
                listener,
                token: None,
                origins: Arc::new(origins),
                hosts: Arc::new(hosts),
                unavailable_reason: Some("feedback requires a loopback listener".into()),
            };
        }
        let token = match std::env::var("LFG_FEEDBACK_TOKEN") {
            Ok(token) if valid_token(&token) => Arc::new(token),
            _ => {
                return Self {
                    store: None,
                    listener,
                    token: None,
                    origins: Arc::new(origins),
                    hosts: Arc::new(hosts),
                    unavailable_reason: Some("feedback requires LFG_FEEDBACK_TOKEN with at least 32 non-whitespace bytes".into()),
                };
            }
        };
        let configured = std::env::var_os("LFG_FEEDBACK_DIR").map(PathBuf::from);
        let (store, unavailable_reason) = match configured {
            None => (
                None,
                Some("feedback storage is disabled; configure LFG_FEEDBACK_DIR".into()),
            ),
            Some(path) => match FeedbackStore::open(path) {
                Ok(store) => (Some(store), None),
                Err(error) => (None, Some(error)),
            },
        };
        Self {
            store,
            listener,
            token: Some(token),
            origins: Arc::new(origins),
            hosts: Arc::new(hosts),
            unavailable_reason,
        }
    }

    pub fn routes(self) -> Router {
        Router::new()
            .route("/api/lfg/feedback/status", get(status))
            .route("/api/lfg/feedback", get(list).post(create))
            .route("/api/lfg/feedback/{feedback_id}", get(read))
            .route("/api/lfg/feedback/{feedback_id}/revisions", post(revise))
            .route("/api/lfg/feedback/{feedback_id}/rollback", post(rollback))
            .with_state(self)
    }
}

impl FeedbackStore {
    fn open(path: PathBuf) -> Result<Self, String> {
        #[cfg(windows)]
        {
            Self::open_windows(path)
        }
        #[cfg(not(windows))]
        {
            Self::open_unix(path)
        }
    }

    #[cfg(windows)]
    fn open_windows(path: PathBuf) -> Result<Self, String> {
        let guard = Arc::new(super::feedback_windows::PrivateDirectory::open(&path)?);
        let db_path = path.join("feedback.sqlite3");
        let db = Connection::open(&db_path)
            .map_err(|e| format!("cannot open private feedback database: {e}"))?;
        let busy_timeout = positive_limit("LFG_FEEDBACK_BUSY_TIMEOUT_MS", 5000)?;
        db.busy_timeout(std::time::Duration::from_millis(busy_timeout))
            .map_err(|e| e.to_string())?;
        db.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS records (
               feedback_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
               created_at TEXT NOT NULL, updated_at TEXT NOT NULL, state TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS revisions (
               feedback_id TEXT NOT NULL REFERENCES records(feedback_id),
               revision INTEGER NOT NULL, parent_revision INTEGER,
               action TEXT NOT NULL, restored_revision INTEGER,
               created_at TEXT NOT NULL, state TEXT NOT NULL,
               PRIMARY KEY(feedback_id, revision)
             );
             CREATE TABLE IF NOT EXISTS feedback_meta (singleton INTEGER PRIMARY KEY CHECK(singleton=1), payload_bytes INTEGER NOT NULL);
             INSERT OR IGNORE INTO feedback_meta(singleton,payload_bytes) SELECT 1,COALESCE(SUM(length(CAST(state AS BLOB))),0) FROM revisions;",
        ).map_err(|e| format!("cannot initialize feedback database: {e}"))?;
        guard.verify_files()?;
        let max_record_bytes = positive_limit(
            "LFG_FEEDBACK_MAX_RECORD_BYTES",
            DEFAULT_MAX_RECORD_BYTES as u64,
        )? as usize;
        let max_journal_bytes =
            positive_limit("LFG_FEEDBACK_MAX_JOURNAL_BYTES", DEFAULT_MAX_JOURNAL_BYTES)?;
        let max_page_size = configured_page_size()?;
        Ok(Self {
            db: Arc::new(Mutex::new(db)),
            _guard: guard,
            max_record_bytes,
            max_journal_bytes,
            max_page_size,
        })
    }

    #[cfg(not(windows))]
    fn open_unix(path: PathBuf) -> Result<Self, String> {
        #[cfg(not(windows))]
        reject_git_location(&path)?;
        let existing_directory = std::fs::symlink_metadata(&path).is_ok();
        if existing_directory {
            verify_private_directory(&path)?;
        } else {
            let parent = path
                .parent()
                .ok_or("feedback directory must have an existing parent")?;
            if !parent.is_dir() {
                return Err("feedback directory parent must already exist".into());
            }
            use std::os::unix::fs::DirBuilderExt;
            let mut builder = std::fs::DirBuilder::new();
            builder.mode(0o700);
            builder
                .create(&path)
                .map_err(|e| format!("cannot create private feedback directory: {e}"))?;
            verify_private_directory(&path)?;
        }
        let path = std::fs::canonicalize(&path)
            .map_err(|e| format!("cannot resolve feedback directory: {e}"))?;
        reject_git_worktree(&path)?;
        verify_safe_ancestors(&path)?;
        let db_path = path.join("feedback.sqlite3");
        ensure_private_file(&db_path)?;
        for sidecar in [
            path.join("feedback.sqlite3-wal"),
            path.join("feedback.sqlite3-shm"),
        ] {
            ensure_private_file(&sidecar)?;
        }
        let db = Connection::open(&db_path)
            .map_err(|e| format!("cannot open private feedback database: {e}"))?;
        let busy_timeout = positive_limit("LFG_FEEDBACK_BUSY_TIMEOUT_MS", 5000)?;
        db.busy_timeout(std::time::Duration::from_millis(busy_timeout))
            .map_err(|e| e.to_string())?;
        db.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS records (
               feedback_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
               created_at TEXT NOT NULL, updated_at TEXT NOT NULL, state TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS revisions (
               feedback_id TEXT NOT NULL REFERENCES records(feedback_id),
               revision INTEGER NOT NULL, parent_revision INTEGER,
               action TEXT NOT NULL, restored_revision INTEGER,
               created_at TEXT NOT NULL, state TEXT NOT NULL,
               PRIMARY KEY(feedback_id, revision)
             );
             CREATE TABLE IF NOT EXISTS feedback_meta (singleton INTEGER PRIMARY KEY CHECK(singleton=1), payload_bytes INTEGER NOT NULL);
             INSERT OR IGNORE INTO feedback_meta(singleton,payload_bytes) SELECT 1,COALESCE(SUM(length(CAST(state AS BLOB))),0) FROM revisions;",
        )
        .map_err(|e| format!("cannot initialize feedback database: {e}"))?;
        for sidecar in [
            db_path.clone(),
            path.join("feedback.sqlite3-wal"),
            path.join("feedback.sqlite3-shm"),
        ] {
            verify_private_file(&sidecar)?;
        }
        let max_record_bytes = positive_limit(
            "LFG_FEEDBACK_MAX_RECORD_BYTES",
            DEFAULT_MAX_RECORD_BYTES as u64,
        )? as usize;
        let max_journal_bytes =
            positive_limit("LFG_FEEDBACK_MAX_JOURNAL_BYTES", DEFAULT_MAX_JOURNAL_BYTES)?;
        let max_page_size = configured_page_size()?;
        Ok(Self {
            db: Arc::new(Mutex::new(db)),
            _guard: (),
            max_record_bytes,
            max_journal_bytes,
            max_page_size,
        })
    }

    fn check_size(&self, state: &Value) -> Result<String, ApiError> {
        if !state.is_object() {
            return Err(ApiError::bad("record state must be a JSON object"));
        }
        if state
            .get("assessmentRef")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return Err(ApiError::bad("assessmentRef is required"));
        }
        if !state.get("provenance").is_some_and(Value::is_object)
            || !state
                .get("originalRecommendation")
                .is_some_and(Value::is_object)
        {
            return Err(ApiError::bad(
                "provenance and originalRecommendation must be JSON objects",
            ));
        }
        let mut state = state.clone();
        if state.get("outcome").is_none_or(Value::is_null) {
            state["outcome"] = json!({"status":"unknown"});
        }
        if state["outcome"].get("status").and_then(Value::as_str) == Some("confirmed") {
            if state["outcome"]
                .get("observation")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
                || state["outcome"]
                    .get("confirmer")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
            {
                return Err(ApiError::bad(
                    "confirmed outcomes require observation and confirmer",
                ));
            }
            // The client supplies a confirmer claim, never an authenticated identity or time.
            state["outcome"]["confirmedAt"] = json!(now()?);
        } else if state["outcome"].get("status").and_then(Value::as_str) != Some("unknown") {
            return Err(ApiError::bad("outcome status must be unknown or confirmed"));
        }
        let bytes =
            serde_json::to_vec(&state).map_err(|_| ApiError::bad("record is not serializable"))?;
        if bytes.len() > self.max_record_bytes {
            return Err(ApiError::too_large(
                "feedback record exceeds configured record limit",
            ));
        }
        Ok(String::from_utf8(bytes).map_err(|_| ApiError::bad("record is not valid UTF-8"))?)
    }
}

#[cfg(not(windows))]
fn reject_git_worktree(path: &Path) -> Result<(), String> {
    if path
        .ancestors()
        .any(|ancestor| ancestor.join(".git").exists())
    {
        return Err("feedback directory is inside a Git worktree".into());
    }
    Ok(())
}

#[cfg(not(windows))]
fn reject_git_location(path: &Path) -> Result<(), String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| format!("cannot resolve feedback path: {e}"))?
            .join(path)
    };
    let mut cursor = absolute.as_path();
    let mut missing = Vec::new();
    while !cursor.exists() {
        let name = cursor
            .file_name()
            .ok_or("feedback path has no existing ancestor")?;
        missing.push(name.to_os_string());
        cursor = cursor
            .parent()
            .ok_or("feedback path has no existing ancestor")?;
    }
    let mut resolved =
        std::fs::canonicalize(cursor).map_err(|e| format!("cannot resolve feedback path: {e}"))?;
    verify_safe_ancestors(&resolved)?;
    for component in missing.iter().rev() {
        resolved.push(component);
    }
    reject_git_worktree(&resolved)
}

#[cfg(unix)]
fn verify_safe_ancestors(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    let uid = unsafe { libc::geteuid() };
    for ancestor in path.ancestors() {
        let metadata = std::fs::metadata(ancestor)
            .map_err(|e| format!("cannot inspect feedback ancestor: {e}"))?;
        if !metadata.is_dir() || (metadata.uid() != uid && metadata.uid() != 0) {
            return Err(
                "feedback path ancestors must be owned by the current account or root".into(),
            );
        }
        if metadata.mode() & 0o022 != 0 && metadata.mode() & 0o1000 == 0 {
            return Err("feedback path has a writable ancestor without the sticky bit".into());
        }
    }
    Ok(())
}

fn local_http_origin(value: &str) -> bool {
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

fn local_authority(value: &str) -> bool {
    if value.contains(['/', '@', '#', '?']) {
        return false;
    }
    let host = value
        .rsplit_once(':')
        .map(|(host, _)| host)
        .unwrap_or(value);
    matches!(host, "localhost" | "127.0.0.1" | "[::1]")
}

fn positive_limit(name: &str, default: u64) -> Result<u64, String> {
    let value = std::env::var(name)
        .ok()
        .map(|v| v.parse::<u64>())
        .transpose()
        .map_err(|_| format!("{name} must be a positive integer"))?
        .unwrap_or(default);
    if value == 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(value)
}

fn configured_page_size() -> Result<usize, String> {
    let value = positive_limit("LFG_FEEDBACK_MAX_PAGE_SIZE", DEFAULT_MAX_PAGE_SIZE as u64)?;
    if value > HARD_MAX_PAGE_SIZE as u64 {
        return Err(format!(
            "LFG_FEEDBACK_MAX_PAGE_SIZE cannot exceed {HARD_MAX_PAGE_SIZE}"
        ));
    }
    Ok(value as usize)
}

fn validate_page_limit(limit: usize, maximum: usize, name: &str) -> Result<(), ApiError> {
    if limit == 0 || limit > maximum {
        return Err(ApiError::bad(format!(
            "{name} must be between 1 and {maximum}"
        )));
    }
    Ok(())
}

fn validate_list_cursor(cursor: Option<&str>) -> Result<(), ApiError> {
    if cursor.is_some_and(|cursor| uuid::Uuid::parse_str(cursor).is_err()) {
        return Err(ApiError::bad(
            "cursor must be a feedbackId returned by this API",
        ));
    }
    Ok(())
}

fn validate_history_before(current_revision: i64, before: Option<i64>) -> Result<(), ApiError> {
    if before.is_some_and(|revision| revision < 1 || revision > current_revision) {
        return Err(ApiError::bad(
            "historyBefore must be a revision number in this record",
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
fn verify_path_not_link(path: &Path) -> Result<(), String> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("cannot inspect feedback path: {error}")),
    };
    if metadata.file_type().is_symlink() {
        return Err("feedback database paths may not be symbolic links".into());
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 {
            return Err("feedback database paths may not be Windows reparse points".into());
        }
    }
    Ok(())
}

#[cfg(unix)]
fn verify_private_directory(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    verify_path_not_link(path)?;
    let metadata =
        std::fs::metadata(path).map_err(|e| format!("cannot inspect feedback directory: {e}"))?;
    if !metadata.is_dir()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
    {
        return Err("existing feedback directory must be owned by this account and have no group or other permissions".into());
    }
    Ok(())
}

#[cfg(unix)]
fn verify_private_file(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    verify_path_not_link(path)?;
    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("cannot inspect feedback database file: {e}"))?;
    if !metadata.is_file()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
        || metadata.nlink() != 1
    {
        return Err("existing feedback database files must be owned by this account, private, and have exactly one hard link".into());
    }
    Ok(())
}
#[cfg(unix)]
fn ensure_private_file(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::OpenOptionsExt;
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
    {
        Ok(file) => {
            drop(file);
            verify_private_file(path)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            verify_private_file(path)
        }
        Err(error) => Err(format!("cannot create private feedback file: {error}")),
    }
}
#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}
impl ApiError {
    fn bad(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_feedback",
            message: message.into(),
        }
    }
    fn too_large(message: &str) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            code: "feedback_too_large",
            message: message.into(),
        }
    }
    fn full(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INSUFFICIENT_STORAGE,
            code: "feedback_storage_full",
            message: message.into(),
        }
    }
    fn internal() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "feedback_storage_error",
            message: "feedback storage operation failed".into(),
        }
    }
    fn unavailable(message: String) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "feedback_unavailable",
            message,
        }
    }
}
impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            [(header::CACHE_CONTROL, "no-store")],
            Json(json!({"error":{"code":self.code,"message":self.message}})),
        )
            .into_response()
    }
}

fn valid_token(token: &str) -> bool {
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

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        difference |= (left.get(index).copied().unwrap_or(0)
            ^ right.get(index).copied().unwrap_or(0)) as usize;
    }
    difference == 0
}

fn authorize(api: &FeedbackApi, peer: SocketAddr, headers: &HeaderMap) -> Result<(), ApiError> {
    if !api.listener.ip().is_loopback() || !peer.ip().is_loopback() {
        return Err(ApiError {
            status: StatusCode::FORBIDDEN,
            code: "local_only",
            message: "feedback is available only to loopback clients".into(),
        });
    }
    let authority = api.listener.to_string();
    let host = headers
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    if host != authority && !api.hosts.iter().any(|allowed| allowed == host) {
        return Err(ApiError {
            status: StatusCode::FORBIDDEN,
            code: "invalid_host",
            message: "Host must match the configured loopback listener".into(),
        });
    }
    if let Some(origin) = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
        let expected = format!("http://{authority}");
        if origin != expected && !api.origins.iter().any(|allowed| allowed == origin) {
            return Err(ApiError {
                status: StatusCode::FORBIDDEN,
                code: "invalid_origin",
                message: "Origin is not allowed for local feedback".into(),
            });
        }
    }
    let Some(token) = &api.token else {
        return Err(ApiError::unavailable(
            api.unavailable_reason
                .clone()
                .unwrap_or_else(|| "feedback is not configured".into()),
        ));
    };
    let supplied = headers
        .get("x-lfg-session-token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !constant_time_eq(supplied.as_bytes(), token.as_bytes()) {
        return Err(ApiError {
            status: StatusCode::FORBIDDEN,
            code: "invalid_session_token",
            message: "a valid local feedback session token is required".into(),
        });
    }
    Ok(())
}

fn store(api: &FeedbackApi) -> Result<&FeedbackStore, ApiError> {
    api.store.as_ref().ok_or_else(|| {
        ApiError::unavailable(
            api.unavailable_reason
                .clone()
                .unwrap_or_else(|| "feedback storage is unavailable".into()),
        )
    })
}

async fn status(
    State(api): State<FeedbackApi>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> Result<([(header::HeaderName, &'static str); 1], Json<Value>), ApiError> {
    authorize(&api, peer, &headers)?;
    Ok((
        [(header::CACHE_CONTROL, "no-store")],
        Json(
            json!({"schemaVersion":"lfg.feedback-status.v1","enabled":api.store.is_some(),"available":api.store.is_some(),"reason":api.unavailable_reason,"defaultPageSize":DEFAULT_PAGE_SIZE,"maxPageSize":api.store.as_ref().map_or(DEFAULT_MAX_PAGE_SIZE, |store| store.max_page_size)}),
        ),
    ))
}

fn now() -> Result<String, ApiError> {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|_| ApiError::internal())
}

fn flattened(
    id: &str,
    revision: i64,
    created: &str,
    updated: &str,
    parent: Option<i64>,
    action: &str,
    restored: Option<i64>,
    state: Value,
) -> Value {
    let mut result = state.as_object().cloned().unwrap_or_default();
    result.insert("feedbackId".into(), json!(id));
    result.insert("revision".into(), json!(revision));
    result.insert("createdAt".into(), json!(created));
    result.insert("updatedAt".into(), json!(updated));
    result.insert("parentRevision".into(), json!(parent));
    result.insert("action".into(), json!(action));
    result.insert("restoredRevision".into(), json!(restored));
    Value::Object(result)
}

async fn create(
    State(api): State<FeedbackApi>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(mut state): Json<Value>,
) -> Result<
    (
        StatusCode,
        [(header::HeaderName, &'static str); 1],
        Json<Value>,
    ),
    ApiError,
> {
    authorize(&api, peer, &headers)?;
    let store = store(&api)?;
    let state_text = store.check_size(&state)?;
    state = serde_json::from_str(&state_text).map_err(|_| ApiError::internal())?;
    let id = uuid::Uuid::new_v4().to_string();
    let time = now()?;
    let mut connection = store.db.lock().unwrap_or_else(|e| e.into_inner());
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| ApiError::internal())?;
    reserve_payload(&tx, state_text.len(), store.max_journal_bytes)?;
    tx.execute("INSERT INTO records(feedback_id,revision,created_at,updated_at,state) VALUES(?1,1,?2,?2,?3)", params![id,time,state_text]).map_err(|_| ApiError::internal())?;
    tx.execute("INSERT INTO revisions(feedback_id,revision,parent_revision,action,restored_revision,created_at,state) VALUES(?1,1,NULL,'create',NULL,?2,?3)", params![id,time,state_text]).map_err(|_| ApiError::internal())?;
    tx.commit().map_err(|_| ApiError::internal())?;
    Ok((
        StatusCode::CREATED,
        [(header::CACHE_CONTROL, "no-store")],
        Json(json!({"record":flattened(&id,1,&time,&time,None,"create",None,state)})),
    ))
}

#[derive(serde::Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ListQuery {
    cursor: Option<String>,
    limit: Option<usize>,
}

async fn list(
    State(api): State<FeedbackApi>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Query(query): Query<ListQuery>,
) -> Result<([(header::HeaderName, &'static str); 1], Json<Value>), ApiError> {
    authorize(&api, peer, &headers)?;
    let store = store(&api)?;
    let limit = query.limit.unwrap_or(DEFAULT_PAGE_SIZE);
    validate_page_limit(limit, store.max_page_size, "limit")?;
    validate_list_cursor(query.cursor.as_deref())?;
    let connection = store.db.lock().unwrap_or_else(|e| e.into_inner());
    let (records, next_cursor) = list_page(&connection, query.cursor.as_deref(), limit)?;
    let has_more = next_cursor.is_some();
    Ok((
        [(header::CACHE_CONTROL, "no-store")],
        Json(
            json!({"schemaVersion":"lfg.feedback-list.v1","records":records,"nextCursor":next_cursor,"hasMore":has_more}),
        ),
    ))
}

fn list_page(
    connection: &Connection,
    cursor: Option<&str>,
    limit: usize,
) -> Result<(Vec<Value>, Option<String>), ApiError> {
    let mut stmt = connection.prepare("SELECT r.feedback_id,r.revision,r.created_at,r.updated_at,r.state,v.parent_revision,v.action,v.restored_revision FROM records r JOIN revisions v ON v.feedback_id=r.feedback_id AND v.revision=r.revision WHERE (?1 IS NULL OR r.feedback_id > ?1) ORDER BY r.feedback_id ASC LIMIT ?2").map_err(|_| ApiError::internal())?;
    let rows = stmt
        .query_map(params![cursor, limit as i64 + 1], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Option<i64>>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, Option<i64>>(7)?,
            ))
        })
        .map_err(|_| ApiError::internal())?;
    let mut rows = rows
        .map(|row| {
            let (id, rev, created, updated, state, parent, action, restored) =
                row.map_err(|_| ApiError::internal())?;
            let state: Value = serde_json::from_str(&state).map_err(|_| ApiError::internal())?;
            Ok(flattened(
                &id, rev, &created, &updated, parent, &action, restored, state,
            ))
        })
        .collect::<Result<Vec<_>, ApiError>>()?;
    let has_more = rows.len() > limit;
    if has_more {
        rows.truncate(limit);
    }
    let next_cursor = has_more
        .then(|| {
            rows.last()
                .and_then(|record| record.get("feedbackId"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .flatten();
    Ok((rows, next_cursor))
}

async fn read(
    State(api): State<FeedbackApi>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<ReadQuery>,
) -> Result<([(header::HeaderName, &'static str); 1], Json<Value>), ApiError> {
    authorize(&api, peer, &headers)?;
    let store = store(&api)?;
    let history_limit = query.history_limit.unwrap_or(DEFAULT_PAGE_SIZE);
    validate_page_limit(history_limit, store.max_page_size, "historyLimit")?;
    let connection = store.db.lock().unwrap_or_else(|e| e.into_inner());
    let current = connection
        .query_row(
            "SELECT r.revision,r.created_at,r.updated_at,r.state,v.parent_revision,v.action,v.restored_revision FROM records r JOIN revisions v ON v.feedback_id=r.feedback_id AND v.revision=r.revision WHERE r.feedback_id=?1",
            [&id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Option<i64>>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<i64>>(6)?,
                ))
            },
        )
        .optional()
        .map_err(|_| ApiError::internal())?
        .ok_or(ApiError {
            status: StatusCode::NOT_FOUND,
            code: "feedback_not_found",
            message: "feedback record not found".into(),
        })?;
    validate_history_before(current.0, query.history_before)?;
    let (history, history_next_before) =
        history_page(&connection, &id, query.history_before, history_limit)?;
    let history_has_more = history_next_before.is_some();
    let state: Value = serde_json::from_str(&current.3).map_err(|_| ApiError::internal())?;
    Ok((
        [(header::CACHE_CONTROL, "no-store")],
        Json(
            json!({"record":flattened(&id,current.0,&current.1,&current.2,current.4,&current.5,current.6,state),"history":history,"historyNextBefore":history_next_before,"historyHasMore":history_has_more}),
        ),
    ))
}

#[derive(serde::Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ReadQuery {
    history_limit: Option<usize>,
    history_before: Option<i64>,
}

fn history_page(
    connection: &Connection,
    id: &str,
    before: Option<i64>,
    limit: usize,
) -> Result<(Vec<Value>, Option<i64>), ApiError> {
    let mut stmt=connection.prepare("SELECT revision,parent_revision,action,restored_revision,created_at,state FROM revisions WHERE feedback_id=?1 AND (?2 IS NULL OR revision < ?2) ORDER BY revision DESC LIMIT ?3").map_err(|_|ApiError::internal())?;
    let rows = stmt
        .query_map(params![id, before, limit as i64 + 1], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, Option<i64>>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, Option<i64>>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
            ))
        })
        .map_err(|_| ApiError::internal())?;
    let mut history=rows.map(|r|{let (revision,parent,action,restored,created,state)=r.map_err(|_|ApiError::internal())?;let state:Value=serde_json::from_str(&state).map_err(|_|ApiError::internal())?;Ok(json!({"revision":revision,"parentRevision":parent,"action":action,"restoredRevision":restored,"createdAt":created,"state":state}))}).collect::<Result<Vec<_>,ApiError>>()?;
    let has_more = history.len() > limit;
    if has_more {
        history.truncate(limit);
    }
    let next_before = has_more
        .then(|| {
            history
                .last()
                .and_then(|entry| entry.get("revision"))
                .and_then(Value::as_i64)
        })
        .flatten();
    Ok((history, next_before))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct RevisionRequest {
    expected_revision: i64,
    state: Value,
}

async fn revise(
    State(api): State<FeedbackApi>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Json(request): Json<RevisionRequest>,
) -> Result<
    (
        StatusCode,
        [(header::HeaderName, &'static str); 1],
        Json<Value>,
    ),
    ApiError,
> {
    authorize(&api, peer, &headers)?;
    append_revision(
        &api,
        &id,
        request.expected_revision,
        request.state,
        None,
        "revise",
    )
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct RollbackRequest {
    expected_revision: i64,
    restore_revision: i64,
}

async fn rollback(
    State(api): State<FeedbackApi>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    Json(request): Json<RollbackRequest>,
) -> Result<
    (
        StatusCode,
        [(header::HeaderName, &'static str); 1],
        Json<Value>,
    ),
    ApiError,
> {
    authorize(&api, peer, &headers)?;
    let store = store(&api)?;
    let mut connection = store.db.lock().unwrap_or_else(|e| e.into_inner());
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| ApiError::internal())?;
    let current = tx
        .query_row(
            "SELECT revision,created_at FROM records WHERE feedback_id=?1",
            [&id],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(|_| ApiError::internal())?
        .ok_or(ApiError {
            status: StatusCode::NOT_FOUND,
            code: "feedback_not_found",
            message: "feedback record not found".into(),
        })?;
    if current.0 != request.expected_revision {
        return Err(conflict(current.0));
    }
    let state_text = tx
        .query_row(
            "SELECT state FROM revisions WHERE feedback_id=?1 AND revision=?2",
            params![id, request.restore_revision],
            |r| r.get::<_, String>(0),
        )
        .optional()
        .map_err(|_| ApiError::internal())?
        .ok_or(ApiError::bad("restoreRevision does not exist"))?;
    if state_text.len() > store.max_record_bytes {
        return Err(ApiError::too_large(
            "restored feedback state exceeds the configured record limit",
        ));
    }
    let time = now()?;
    let revision = current.0 + 1;
    reserve_payload(&tx, state_text.len(), store.max_journal_bytes)?;
    tx.execute("INSERT INTO revisions(feedback_id,revision,parent_revision,action,restored_revision,created_at,state) VALUES(?1,?2,?3,'rollback',?4,?5,?6)",params![id,revision,current.0,request.restore_revision,time,state_text]).map_err(|_|ApiError::internal())?;
    tx.execute(
        "UPDATE records SET revision=?2,updated_at=?3,state=?4 WHERE feedback_id=?1",
        params![id, revision, time, state_text],
    )
    .map_err(|_| ApiError::internal())?;
    tx.commit().map_err(|_| ApiError::internal())?;
    let state: Value = serde_json::from_str(&state_text).map_err(|_| ApiError::internal())?;
    Ok((
        StatusCode::CREATED,
        [(header::CACHE_CONTROL, "no-store")],
        Json(
            json!({"record":flattened(&id,revision,&current.1,&time,Some(current.0),"rollback",Some(request.restore_revision),state)}),
        ),
    ))
}

fn append_revision(
    api: &FeedbackApi,
    id: &str,
    expected: i64,
    state: Value,
    restored: Option<i64>,
    action: &str,
) -> Result<
    (
        StatusCode,
        [(header::HeaderName, &'static str); 1],
        Json<Value>,
    ),
    ApiError,
> {
    let store = store(api)?;
    let state_text = store.check_size(&state)?;
    let mut connection = store.db.lock().unwrap_or_else(|e| e.into_inner());
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| ApiError::internal())?;
    let current = tx
        .query_row(
            "SELECT revision,created_at FROM records WHERE feedback_id=?1",
            [id],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(|_| ApiError::internal())?
        .ok_or(ApiError {
            status: StatusCode::NOT_FOUND,
            code: "feedback_not_found",
            message: "feedback record not found".into(),
        })?;
    if current.0 != expected {
        return Err(conflict(current.0));
    }
    reserve_payload(&tx, state_text.len(), store.max_journal_bytes)?;
    let time = now()?;
    let revision = current.0 + 1;
    tx.execute("INSERT INTO revisions(feedback_id,revision,parent_revision,action,restored_revision,created_at,state) VALUES(?1,?2,?3,?4,?5,?6,?7)",params![id,revision,current.0,action,restored,time,state_text]).map_err(|_|ApiError::internal())?;
    tx.execute(
        "UPDATE records SET revision=?2,updated_at=?3,state=?4 WHERE feedback_id=?1",
        params![id, revision, time, state_text],
    )
    .map_err(|_| ApiError::internal())?;
    tx.commit().map_err(|_| ApiError::internal())?;
    let normalized: Value = serde_json::from_str(&state_text).map_err(|_| ApiError::internal())?;
    Ok((
        StatusCode::CREATED,
        [(header::CACHE_CONTROL, "no-store")],
        Json(
            json!({"record":flattened(id,revision,&current.1,&time,Some(current.0),action,restored,normalized)}),
        ),
    ))
}
fn conflict(current: i64) -> ApiError {
    ApiError {
        status: StatusCode::CONFLICT,
        code: "revision_conflict",
        message: format!("expected revision did not match current revision {current}"),
    }
}

fn reserve_payload(
    tx: &rusqlite::Transaction<'_>,
    added_bytes: usize,
    limit: u64,
) -> Result<(), ApiError> {
    let current: i64 = tx
        .query_row(
            "SELECT payload_bytes FROM feedback_meta WHERE singleton=1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| ApiError::internal())?;
    let next = u64::try_from(current)
        .ok()
        .and_then(|value| value.checked_add(added_bytes as u64))
        .ok_or_else(|| ApiError::full("feedback logical history byte count overflowed"))?;
    if next > limit || next > i64::MAX as u64 {
        return Err(ApiError::full(format!(
            "feedback logical revision history limit reached ({current} of {limit} bytes); archive or configure retention explicitly"
        )));
    }
    tx.execute(
        "UPDATE feedback_meta SET payload_bytes=?1 WHERE singleton=1",
        [next as i64],
    )
    .map_err(|_| ApiError::internal())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    fn state(label: &str) -> Value {
        json!({"assessmentRef":"assess-1","provenance":{"engineId":"deterministic","sourceRef":"sha256:abc123","extra":"kept"},"originalRecommendation":{"intent":"build"},"ownerCorrection":{"intent":label,"teacherProposal":{"x":1}},"outcome":{"status":"unknown","note":"pending"},"custom":{"preserve":true}})
    }
    #[test]
    fn create_revise_conflict_rollback_and_restart_preserve_history_and_extensions() {
        let d = tempfile::tempdir().unwrap();
        let store = FeedbackStore::open(d.path().join("private")).unwrap();
        let initial = state("corrected");
        let id = uuid::Uuid::new_v4().to_string();
        let text = store.check_size(&initial).unwrap();
        let c = store.db.lock().unwrap();
        let tx = c.unchecked_transaction().unwrap();
        let at = "2026-01-01T00:00:00Z";
        tx.execute(
            "INSERT INTO records VALUES(?1,1,?2,?2,?3)",
            params![id, at, text],
        )
        .unwrap();
        tx.execute(
            "INSERT INTO revisions VALUES(?1,1,NULL,'create',NULL,?2,?3)",
            params![id, at, text],
        )
        .unwrap();
        tx.commit().unwrap();
        drop(c);
        let appended = append_revision(
            &FeedbackApi {
                store: Some(store.clone()),
                listener: "127.0.0.1:9000".parse().unwrap(),
                token: Some(Arc::new("abcdefghijklmnopqrstuvwxyz012345".into())),
                origins: Arc::new(vec![]),
                hosts: Arc::new(vec![]),
                unavailable_reason: None,
            },
            &id,
            1,
            state("revised"),
            None,
            "revise",
        )
        .unwrap();
        assert_eq!(
            appended.2.0["record"]["provenance"]["sourceRef"],
            "sha256:abc123"
        );
        assert_eq!(appended.2.0["record"]["outcome"]["status"], "unknown");
        assert_eq!(
            append_revision(
                &FeedbackApi {
                    store: Some(store.clone()),
                    listener: "127.0.0.1:9000".parse().unwrap(),
                    token: Some(Arc::new("abcdefghijklmnopqrstuvwxyz012345".into())),
                    origins: Arc::new(vec![]),
                    hosts: Arc::new(vec![]),
                    unavailable_reason: None
                },
                &id,
                1,
                state("lost"),
                None,
                "revise"
            )
            .unwrap_err()
            .status,
            StatusCode::CONFLICT
        );
        drop(store);
        let store = FeedbackStore::open(d.path().join("private")).unwrap();
        let c = store.db.lock().unwrap();
        let count: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM revisions WHERE feedback_id=?1",
                [id.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        let current: String = c
            .query_row(
                "SELECT state FROM records WHERE feedback_id=?1",
                [id.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        let current: Value = serde_json::from_str(&current).unwrap();
        assert_eq!(count, 2);
        assert_eq!(current["ownerCorrection"]["intent"], "revised");
        assert_eq!(current["custom"]["preserve"], true);
    }
    #[cfg(not(windows))]
    #[test]
    fn defaults_unknown_and_rejects_git_worktrees_and_size_overflow() {
        let d = tempfile::tempdir().unwrap();
        let store = FeedbackStore::open(d.path().join("private")).unwrap();
        let mut s = state("ok");
        s["outcome"] = Value::Null;
        let normalized: Value = serde_json::from_str(&store.check_size(&s).unwrap()).unwrap();
        assert_eq!(normalized["outcome"]["status"], "unknown");
        assert!(store.check_size(&json!({"assessmentRef":"x","provenance":{},"originalRecommendation":{},"big":"x".repeat(DEFAULT_MAX_RECORD_BYTES)})).is_err());
        let root = d.path().join("worktree");
        std::fs::create_dir_all(root.join("nested")).unwrap();
        std::fs::write(root.join(".git"), "gitdir: x").unwrap();
        assert!(reject_git_worktree(&root.join("nested")).is_err());
        assert!(reject_git_location(&root.join("new-private-dir")).is_err());
        let unsafe_parent = d.path().join("writable-parent");
        std::fs::create_dir(&unsafe_parent).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&unsafe_parent, std::fs::Permissions::from_mode(0o777)).unwrap();
        let unsafe_target = unsafe_parent.join("private");
        assert!(reject_git_location(&unsafe_target).is_err());
        assert!(!unsafe_target.exists());
        #[cfg(unix)]
        {
            let link = d.path().join("linked-worktree");
            std::os::unix::fs::symlink(&root, &link).unwrap();
            assert!(reject_git_location(&link.join("private")).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn refuses_existing_database_symlink_before_opening_it() {
        let d = tempfile::tempdir().unwrap();
        let private = d.path().join("private");
        let store = FeedbackStore::open(private.clone()).unwrap();
        drop(store);
        let db = private.join("feedback.sqlite3");
        std::fs::remove_file(&db).unwrap();
        let external = d.path().join("outside.sqlite3");
        std::fs::write(&external, b"not a database").unwrap();
        std::fs::set_permissions(&external, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::os::unix::fs::symlink(&external, &db).unwrap();
        assert!(FeedbackStore::open(private).is_err());
        assert_eq!(std::fs::read(external).unwrap(), b"not a database");
    }

    #[cfg(unix)]
    #[test]
    fn refuses_existing_database_hardlink_before_opening_it() {
        use std::os::unix::fs::PermissionsExt;
        let d = tempfile::tempdir().unwrap();
        let private = d.path().join("private");
        std::fs::create_dir(&private).unwrap();
        std::fs::set_permissions(&private, std::fs::Permissions::from_mode(0o700)).unwrap();
        let external = d.path().join("outside.sqlite3");
        std::fs::write(&external, b"not a database").unwrap();
        std::fs::set_permissions(&external, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::hard_link(&external, private.join("feedback.sqlite3")).unwrap();
        assert!(FeedbackStore::open(private).is_err());
        assert_eq!(std::fs::read(external).unwrap(), b"not a database");
    }

    #[test]
    fn confirmed_outcomes_get_server_time_and_require_observation_and_confirmer() {
        let d = tempfile::tempdir().unwrap();
        let store = FeedbackStore::open(d.path().join("private")).unwrap();
        let mut s = state("ok");
        s["outcome"] = json!({"status":"confirmed","observation":"done","confirmer":"local operator","confirmedAt":"caller-supplied"});
        let saved: Value = serde_json::from_str(&store.check_size(&s).unwrap()).unwrap();
        assert_ne!(saved["outcome"]["confirmedAt"], "caller-supplied");
        assert!(
            saved["outcome"]["confirmedAt"]
                .as_str()
                .unwrap()
                .ends_with('Z')
        );
        s["outcome"]["confirmer"] = Value::Null;
        assert!(store.check_size(&s).is_err());
    }

    #[test]
    fn every_route_rejects_missing_or_wrong_secret_and_keeps_loopback_origin_checks() {
        let api = FeedbackApi {
            store: None,
            listener: "127.0.0.1:9010".parse().unwrap(),
            token: Some(Arc::new("abcdefghijklmnopqrstuvwxyz012345".into())),
            origins: Arc::new(vec!["http://localhost:3000".into()]),
            hosts: Arc::new(vec!["localhost:3000".into()]),
            unavailable_reason: None,
        };
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, "127.0.0.1:9010".parse().unwrap());
        assert!(authorize(&api, "127.0.0.1:7777".parse().unwrap(), &headers).is_err());
        headers.insert("x-lfg-session-token", "wrong-secret".parse().unwrap());
        assert!(authorize(&api, "127.0.0.1:7777".parse().unwrap(), &headers).is_err());
        headers.insert(
            "x-lfg-session-token",
            "abcdefghijklmnopqrstuvwxyz012345".parse().unwrap(),
        );
        assert!(authorize(&api, "127.0.0.1:7777".parse().unwrap(), &headers).is_ok());
        assert!(authorize(&api, "192.168.1.2:7777".parse().unwrap(), &headers).is_err());
        headers.insert(header::HOST, "localhost:9010".parse().unwrap());
        assert!(authorize(&api, "127.0.0.1:7777".parse().unwrap(), &headers).is_err());
        headers.insert(header::HOST, "127.0.0.1:9010".parse().unwrap());
        headers.insert(header::ORIGIN, "https://attacker.invalid".parse().unwrap());
        assert!(authorize(&api, "127.0.0.1:7777".parse().unwrap(), &headers).is_err());
        headers.remove(header::ORIGIN);
        headers.insert(
            "x-lfg-session-token",
            "abcdefghijklmnopqrstuvwxyz012345".parse().unwrap(),
        );
        assert!(authorize(&api, "127.0.0.1:7777".parse().unwrap(), &headers).is_ok());
        assert!(valid_token("abcdefghijklmnopqrstuvwxyz012345"));
        assert!(!valid_token("short"));
        assert!(!valid_token("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    }

    #[test]
    fn logical_history_limit_counts_each_retained_revision_transactionally() {
        let dir = tempfile::tempdir().unwrap();
        let store = FeedbackStore::open(dir.path().join("private")).unwrap();
        let connection = store.db.lock().unwrap();
        let mut connection = connection;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        reserve_payload(&tx, 80, 150).unwrap();
        tx.commit().unwrap();
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        assert_eq!(
            reserve_payload(&tx, 71, 150).unwrap_err().status,
            StatusCode::INSUFFICIENT_STORAGE
        );
        drop(tx);
        let stored: i64 = connection
            .query_row(
                "SELECT payload_bytes FROM feedback_meta WHERE singleton=1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stored, 80);
    }

    #[test]
    fn simultaneous_writers_cannot_overbook_logical_history_limit() {
        let dir = tempfile::tempdir().unwrap();
        let store = FeedbackStore::open(dir.path().join("private")).unwrap();
        let db_path = dir.path().join("private").join("feedback.sqlite3");
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let threads = (0..2)
            .map(|_| {
                let barrier = barrier.clone();
                let db_path = db_path.clone();
                std::thread::spawn(move || {
                    let mut connection = Connection::open(db_path).unwrap();
                    barrier.wait();
                    let tx = connection
                        .transaction_with_behavior(TransactionBehavior::Immediate)
                        .unwrap();
                    match reserve_payload(&tx, 100, 150) {
                        Ok(()) => {
                            tx.commit().unwrap();
                            true
                        }
                        Err(_) => false,
                    }
                })
            })
            .collect::<Vec<_>>();
        let accepted = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .filter(|accepted| *accepted)
            .count();
        assert_eq!(accepted, 1);
        let c = store.db.lock().unwrap();
        let bytes: i64 = c
            .query_row(
                "SELECT payload_bytes FROM feedback_meta WHERE singleton=1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(bytes, 100);
    }

    #[test]
    fn list_pages_are_stable_by_feedback_id_and_preserve_extensions() {
        let dir = tempfile::tempdir().unwrap();
        let store = FeedbackStore::open(dir.path().join("private")).unwrap();
        let connection = store.db.lock().unwrap();
        let ids = [
            "00000000-0000-4000-8000-000000000001",
            "00000000-0000-4000-8000-000000000002",
            "00000000-0000-4000-8000-000000000003",
        ];
        for (index, id) in ids.iter().enumerate() {
            let state = json!({"assessmentRef":format!("a{index}"),"provenance":{"engineId":"test"},"originalRecommendation":{"intent":"build"},"ownerCorrection":null,"outcome":{"status":"unknown"},"extension":{"keep":index}});
            let state = serde_json::to_string(&state).unwrap();
            let at = format!("2026-01-01T00:00:0{index}Z");
            connection
                .execute(
                    "INSERT INTO records VALUES(?1,1,?2,?2,?3)",
                    params![id, at, state],
                )
                .unwrap();
            connection
                .execute(
                    "INSERT INTO revisions VALUES(?1,1,NULL,'create',NULL,?2,?3)",
                    params![id, at, state],
                )
                .unwrap();
        }
        let (first, cursor) = list_page(&connection, None, 2).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first[0]["feedbackId"], ids[0]);
        assert_eq!(first[1]["feedbackId"], ids[1]);
        assert_eq!(first[1]["extension"]["keep"], 1);
        let cursor = cursor.unwrap();
        // Changing the timestamp of a later row cannot move it across this cursor.
        connection
            .execute(
                "UPDATE records SET updated_at='2027-01-01T00:00:00Z' WHERE feedback_id=?1",
                [ids[2]],
            )
            .unwrap();
        let (second, next) = list_page(&connection, Some(&cursor), 2).unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0]["feedbackId"], ids[2]);
        assert!(next.is_none());
    }

    #[test]
    fn history_pages_are_latest_first_and_accept_only_valid_cursors() {
        let dir = tempfile::tempdir().unwrap();
        let store = FeedbackStore::open(dir.path().join("private")).unwrap();
        let connection = store.db.lock().unwrap();
        let id = "00000000-0000-4000-8000-000000000099";
        for revision in 1..=5i64 {
            let state = json!({"assessmentRef":"a","provenance":{"engineId":"test"},"originalRecommendation":{"intent":"build"},"outcome":{"status":"unknown"},"extension":{"revision":revision}});
            let state = serde_json::to_string(&state).unwrap();
            let at = format!("2026-01-01T00:00:0{}Z", revision);
            if revision == 1 {
                connection
                    .execute(
                        "INSERT INTO records VALUES(?1,1,?2,?2,?3)",
                        params![id, at, state],
                    )
                    .unwrap();
            } else {
                connection.execute("UPDATE records SET revision=?2,updated_at=?3,state=?4 WHERE feedback_id=?1", params![id,revision,at,state]).unwrap();
            }
            connection
                .execute(
                    "INSERT INTO revisions VALUES(?1,?2,?3,?4,NULL,?5,?6)",
                    params![
                        id,
                        revision,
                        (revision > 1).then_some(revision - 1),
                        if revision == 1 { "create" } else { "revise" },
                        at,
                        state
                    ],
                )
                .unwrap();
        }
        validate_page_limit(2, 3, "historyLimit").unwrap();
        validate_history_before(5, Some(2)).unwrap();
        assert!(validate_page_limit(4, 3, "historyLimit").is_err());
        assert!(validate_page_limit(0, 3, "historyLimit").is_err());
        assert!(validate_history_before(5, Some(0)).is_err());
        assert!(validate_history_before(5, Some(6)).is_err());
        assert!(validate_list_cursor(Some("not-a-uuid")).is_err());

        let (latest, before) = history_page(&connection, id, None, 2).unwrap();
        assert_eq!(
            latest
                .iter()
                .map(|item| item["revision"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            [5, 4]
        );
        assert_eq!(latest[0]["state"]["extension"]["revision"], 5);
        assert_eq!(before, Some(4));
        let (older, before) = history_page(&connection, id, before, 2).unwrap();
        assert_eq!(
            older
                .iter()
                .map(|item| item["revision"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            [3, 2]
        );
        assert_eq!(before, Some(2));
        let (oldest, before) = history_page(&connection, id, before, 2).unwrap();
        assert_eq!(oldest[0]["revision"], 1);
        assert!(before.is_none());
    }
}
