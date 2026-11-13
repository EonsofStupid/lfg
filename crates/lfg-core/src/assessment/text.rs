//! Text handling shared by every job: what is scored, whole-word matching and
//! task clauses.

use super::config::AssessmentConfig;

/// The text the rules and the model assess: the configured fields of a JSON
/// object when present, otherwise the text as given.
pub fn scored_text<'a>(config: &AssessmentConfig, text: &'a str) -> std::borrow::Cow<'a, str> {
    let Some(structured) = &config.structured_input else {
        return std::borrow::Cow::Borrowed(text);
    };
    if !text.trim_start().starts_with('{') {
        return std::borrow::Cow::Borrowed(text);
    }
    let Ok(serde_json::Value::Object(object)) = serde_json::from_str::<serde_json::Value>(text)
    else {
        return std::borrow::Cow::Borrowed(text);
    };
    let parts: Vec<&str> = structured
        .json_fields
        .iter()
        .filter_map(|field| object.get(field).and_then(serde_json::Value::as_str))
        .filter(|value| !value.trim().is_empty())
        .collect();
    if parts.is_empty() {
        std::borrow::Cow::Borrowed(text)
    } else {
        std::borrow::Cow::Owned(parts.join(&structured.separator))
    }
}

/// True when `word` appears as a whole word or phrase, allowing a configured
/// ending such as `tests` for `test`, or `updating` for `update`.
pub fn contains_word(text: &str, word: &str, suffixes: &[String]) -> bool {
    let ends_cleanly = |rest: &str, allowed: &dyn Fn(&str) -> bool| {
        let token_end = rest
            .find(|c: char| !c.is_alphanumeric())
            .unwrap_or(rest.len());
        let token = &rest[..token_end];
        token.is_empty() || allowed(token)
    };
    let starts_cleanly = |start: usize| {
        !text[..start]
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric)
    };
    let search = |needle: &str, allowed: &dyn Fn(&str) -> bool| {
        text.match_indices(needle).any(|(start, _)| {
            starts_cleanly(start) && ends_cleanly(&text[start + needle.len()..], allowed)
        })
    };
    if search(word, &|token| suffixes.iter().any(|s| s == token)) {
        return true;
    }
    match word.strip_suffix('e') {
        Some(stem) if !stem.is_empty() && suffixes.iter().any(|s| s == "ing") => {
            text.match_indices(stem).any(|(start, _)| {
                starts_cleanly(start) && {
                    let rest = &text[start + stem.len()..];
                    rest.strip_prefix("ing")
                        .is_some_and(|after| ends_cleanly(after, &|_| false))
                }
            })
        }
        _ => false,
    }
}

pub(super) fn matched_words(text: &str, words: &[String], suffixes: &[String]) -> Vec<String> {
    words
        .iter()
        .filter(|word| contains_word(text, word, suffixes))
        .cloned()
        .collect()
}

pub(super) fn scored_slice(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

pub(super) fn task_clauses(normalized: &str, config: &AssessmentConfig) -> Vec<String> {
    let mut clauses = vec![normalized.to_string()];
    for separator in &config.task_separators {
        clauses = clauses
            .iter()
            .flat_map(|clause| clause.split(separator.as_str()))
            .map(str::to_string)
            .collect();
    }
    clauses
        .into_iter()
        .map(|clause| clause.trim().trim_start_matches("and ").trim().to_string())
        .filter(|clause| {
            config
                .action_words
                .iter()
                .any(|word| contains_word(clause, word, &config.word_suffixes))
        })
        .map(|clause| {
            if clause.chars().count() > config.max_task_chars {
                let short: String = clause.chars().take(config.max_task_chars).collect();
                format!("{short}…")
            } else {
                clause
            }
        })
        .collect()
}
