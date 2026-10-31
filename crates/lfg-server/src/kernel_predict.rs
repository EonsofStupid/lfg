use crate::kernel_journal::{Journal, JsonlJournal};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    io::{self, BufRead, Read, Write},
    path::PathBuf,
};

#[derive(Deserialize)]
struct Example {
    id: String,
    text: String,
}

#[derive(Serialize)]
struct Prediction<'a> {
    schema_version: &'static str,
    id: &'a str,
    input_sha256: String,
    result: lfg_core::kernel::ResultRecord,
    #[serde(skip_serializing_if = "Option::is_none")]
    journal_sequence: Option<u64>,
}

pub fn run(
    snapshot: Option<PathBuf>,
    max_examples: usize,
    journal: Option<PathBuf>,
    journal_max_bytes: Option<u64>,
    capture_text: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if max_examples == 0 {
        return Err("max examples must be positive".into());
    }
    let operations = lfg_server::configured_operations()?;
    let engine = crate::load_kernel(snapshot)?;
    let mut journal = match (journal, journal_max_bytes) {
        (None, None) if !capture_text => None,
        (Some(path), Some(limit)) => Some(JsonlJournal::open(&path, limit, true)?),
        _ => return Err("journal path and byte limit must be provided together; text capture requires a journal".into()),
    };
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let mut seen = HashSet::new();
    let mut line = String::new();
    loop {
        line.clear();
        let read = reader
            .by_ref()
            .take((operations.max_http_body_bytes() as u64).saturating_add(1))
            .read_line(&mut line)?;
        if read == 0 {
            break;
        }
        if read > operations.max_http_body_bytes() {
            return Err("input line exceeds configured byte limit".into());
        }
        if line.trim().is_empty() {
            continue;
        }
        let row: Example = serde_json::from_str(&line)?;
        if seen.len() >= max_examples || row.id.trim().is_empty() || !seen.insert(row.id.clone()) {
            return Err("example limit exceeded or ID is empty or duplicated".into());
        }
        operations.validate_prompt("kernel-evaluation", &row.text)?;
        let result = engine.evaluate(&BTreeMap::from([("prompt".into(), row.text.clone())]))?;
        let journal_sequence = journal
            .as_mut()
            .map(|store| store.observe(&row.id, &row.text, &result, capture_text))
            .transpose()?;
        serde_json::to_writer(
            &mut output,
            &Prediction {
                schema_version: "lfg.kernel-prediction.v1",
                id: &row.id,
                input_sha256: lfg_core::digest_text(&row.text),
                result,
                journal_sequence,
            },
        )?;
        writeln!(output)?;
        output.flush()?;
    }
    if seen.is_empty() {
        return Err("no examples supplied".into());
    }
    Ok(())
}
