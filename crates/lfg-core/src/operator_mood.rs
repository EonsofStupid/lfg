//! Reads the operator's mood from a prompt with configured patterns, so an
//! agent can be told when to answer with grounded facts instead of
//! reassurance. The patterns are a word list: a reading is a claim, not a
//! verified label, and it never stores or repeats the prompt's words.

use std::collections::BTreeSet;

use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};

use crate::digest_value;

pub const SCHEMA_VERSION: &str = "lfg.operator-mood.v1";
const MAX_GUIDANCE_CHARS: usize = 10_000;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MoodPattern {
    pub id: String,
    pub pattern: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CapsBurst {
    /// How many all-caps words, outside `ignore`, count as shouting.
    pub min_words: usize,
    pub min_length: usize,
    pub ignore: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Guidance {
    /// Context added for the agent when a prompt reads as frustrated.
    pub frustrated: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OperatorMoodFile {
    pub schema_version: String,
    pub revision: String,
    pub max_text_bytes: usize,
    pub frustration: Vec<MoodPattern>,
    pub positive: Vec<MoodPattern>,
    pub caps_burst: CapsBurst,
    pub guidance: Guidance,
}

#[derive(Clone, Debug)]
pub struct OperatorMood {
    file: OperatorMoodFile,
    frustration: Vec<(String, Regex)>,
    positive: Vec<(String, Regex)>,
    caps: Regex,
    ignore: BTreeSet<String>,
}

/// One reading of a prompt: `frustrated`, `positive` or `neutral`, with the
/// configured pattern ids that fired. Frustration wins over praise, since
/// praise words are often sarcastic in an angry message.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MoodReading {
    pub state: &'static str,
    pub signals: Vec<String>,
}

fn compile(
    patterns: &[MoodPattern],
    seen: &mut BTreeSet<String>,
) -> Result<Vec<(String, Regex)>, String> {
    patterns
        .iter()
        .map(|item| {
            let id_ok = !item.id.is_empty()
                && item.id.len() <= 64
                && item
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
            if !id_ok || !seen.insert(item.id.clone()) || item.pattern.trim().is_empty() {
                return Err(format!(
                    "invalid or duplicate mood pattern id {:?}",
                    item.id
                ));
            }
            RegexBuilder::new(&item.pattern)
                .case_insensitive(true)
                .size_limit(1 << 20)
                .build()
                .map(|regex| (item.id.clone(), regex))
                .map_err(|_| format!("mood pattern {} does not compile", item.id))
        })
        .collect()
}

impl OperatorMood {
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let file: OperatorMoodFile = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if file.schema_version != SCHEMA_VERSION
            || file.revision.trim().is_empty()
            || file.max_text_bytes == 0
            || file.frustration.is_empty()
            || file.caps_burst.min_words == 0
            || file.caps_burst.min_length < 2
            || file.caps_burst.min_length > 32
            || file.guidance.frustrated.trim().is_empty()
            || file.guidance.frustrated.chars().count() > MAX_GUIDANCE_CHARS
        {
            return Err("unsupported operator mood configuration".into());
        }
        let mut seen = BTreeSet::new();
        let frustration = compile(&file.frustration, &mut seen)?;
        let positive = compile(&file.positive, &mut seen)?;
        let caps = Regex::new(&format!(r"\b[A-Z]{{{},}}\b", file.caps_burst.min_length))
            .map_err(|e| e.to_string())?;
        let ignore = file.caps_burst.ignore.iter().cloned().collect();
        Ok(Self {
            file,
            frustration,
            positive,
            caps,
            ignore,
        })
    }

    pub fn bundled() -> Self {
        Self::from_json_str(include_str!("../config/operator-mood.v1.json"))
            .expect("bundled operator mood configuration is valid")
    }

    pub fn revision(&self) -> &str {
        &self.file.revision
    }

    pub fn digest(&self) -> String {
        digest_value(&serde_json::to_value(&self.file).expect("mood configuration serializes"))
    }

    pub fn read(&self, text: &str) -> MoodReading {
        let mut end = text.len().min(self.file.max_text_bytes);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let text = &text[..end];
        let mut signals: Vec<String> = self
            .frustration
            .iter()
            .filter(|(_, regex)| regex.is_match(text))
            .map(|(id, _)| id.clone())
            .collect();
        let shouted = self
            .caps
            .find_iter(text)
            .filter(|word| !self.ignore.contains(word.as_str()))
            .count();
        if shouted >= self.file.caps_burst.min_words {
            signals.push("caps-burst".into());
        }
        if !signals.is_empty() {
            return MoodReading {
                state: "frustrated",
                signals,
            };
        }
        let praise: Vec<String> = self
            .positive
            .iter()
            .filter(|(_, regex)| regex.is_match(text))
            .map(|(id, _)| id.clone())
            .collect();
        if praise.is_empty() {
            MoodReading {
                state: "neutral",
                signals: Vec::new(),
            }
        } else {
            MoodReading {
                state: "positive",
                signals: praise,
            }
        }
    }

    /// The context to add for the agent, if any, for this reading.
    pub fn guidance(&self, reading: &MoodReading) -> Option<&str> {
        (reading.state == "frustrated").then_some(self.file.guidance.frustrated.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustration_wins_and_reports_pattern_ids_not_words() {
        let mood = OperatorMood::bundled();
        let reading = mood.read("why are you not using the fucking sparks?");
        assert_eq!(reading.state, "frustrated");
        assert_eq!(reading.signals, vec!["profanity", "why-you"]);
        assert!(mood.guidance(&reading).is_some());
        // Praise inside an angry message is treated as sarcasm.
        let sarcastic = mood.read("great, it's STILL BROKEN after all of that!!");
        assert_eq!(sarcastic.state, "frustrated");
        assert!(sarcastic.signals.contains(&"caps-burst".to_string()));
        assert!(
            sarcastic
                .signals
                .contains(&"repeated-punctuation".to_string())
        );
    }

    #[test]
    fn calm_and_positive_messages_add_nothing() {
        let mood = OperatorMood::bundled();
        let calm = mood.read("Stop the dev server and rerun the UI tests.");
        assert_eq!(calm.state, "neutral");
        assert!(mood.guidance(&calm).is_none());
        let pleased = mood.read("Perfect, thank you. That works now.");
        assert_eq!(pleased.state, "positive");
        assert!(mood.guidance(&pleased).is_none());
        // Common acronyms are not shouting.
        assert_eq!(
            mood.read("Check the CI and the API JSON for the PR").state,
            "neutral"
        );
    }

    #[test]
    fn configuration_is_validated() {
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../config/operator-mood.v1.json")).unwrap();
        assert!(OperatorMood::from_json_str(&value.to_string()).is_ok());
        value["frustration"][1]["id"] = serde_json::json!("profanity");
        assert!(OperatorMood::from_json_str(&value.to_string()).is_err());
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../config/operator-mood.v1.json")).unwrap();
        value["frustration"][0]["pattern"] = serde_json::json!("(unclosed");
        assert!(OperatorMood::from_json_str(&value.to_string()).is_err());
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../config/operator-mood.v1.json")).unwrap();
        value["guidance"]["frustrated"] = serde_json::json!("x".repeat(10_001));
        assert!(OperatorMood::from_json_str(&value.to_string()).is_err());
        value["guidance"]["frustrated"] = serde_json::json!("ok");
        value["extra"] = serde_json::json!(true);
        assert!(OperatorMood::from_json_str(&value.to_string()).is_err());
    }

    #[test]
    fn long_text_is_cut_at_a_character_boundary() {
        let mood = OperatorMood::bundled();
        let text = format!("{}wtf", "é".repeat(40_000));
        // The signal lies past the configured limit, so it is not read.
        assert_eq!(mood.read(&text).state, "neutral");
    }
}
