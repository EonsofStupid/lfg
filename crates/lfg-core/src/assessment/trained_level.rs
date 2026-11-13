//! The trained work-level model: its binding in the configuration, one run's
//! output, and what it contributed to the complexity bracket.

use serde::{Deserialize, Serialize};

use super::config::Bracket;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelScoreBinding {
    pub id: String,
    /// SHA-256 of the model directory's `manifest.json`, which pins every file.
    pub manifest_sha256: String,
    /// The bracket for each model level, lowest first.
    pub level_brackets: Vec<String>,
    /// Ascending expected-level thresholds: the level is the number reached.
    pub level_cut_points: Vec<f64>,
}

/// One run of the trained model: probabilities for each level, lowest first.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelOutput {
    pub probabilities: Vec<f64>,
    pub elapsed_micros: u64,
    /// True when the prompt was longer than the model reads.
    pub truncated: bool,
}

/// What the trained model contributed, reported beside the rule result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelScore {
    /// `available`, `unavailable` or `invalid_output`.
    pub status: String,
    pub id: String,
    pub manifest_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_level: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub probabilities: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<usize>,
    /// The bracket the model chose, which the assessment then uses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bracket: Option<String>,
    /// The bracket the rules alone would have chosen.
    pub rules_bracket: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_micros: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncated: Option<bool>,
}

impl ModelScoreBinding {
    pub(super) fn validate(&self, brackets: &[Bracket]) -> Result<(), String> {
        let digest_ok = self.manifest_sha256.len() == 64
            && self
                .manifest_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        let id_ok = !self.id.is_empty()
            && self.id.len() <= 128
            && self
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b));
        let known = |id: &String| brackets.iter().any(|bracket| &bracket.id == id);
        let ascending = self
            .level_cut_points
            .windows(2)
            .all(|pair| pair[0] < pair[1]);
        if !digest_ok
            || !id_ok
            || self.level_brackets.len() < 2
            || !self.level_brackets.iter().all(known)
            || self.level_cut_points.len() + 1 != self.level_brackets.len()
            || self.level_cut_points.iter().any(|cut| !cut.is_finite())
            || !ascending
        {
            return Err("invalid model score binding".into());
        }
        Ok(())
    }

    /// Expected level, level and bracket index for valid probabilities.
    pub(super) fn choose(
        &self,
        brackets: &[Bracket],
        probabilities: &[f64],
    ) -> Option<(f64, usize, usize)> {
        let total: f64 = probabilities.iter().sum();
        if probabilities.len() != self.level_brackets.len()
            || probabilities
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || (total - 1.0).abs() > 1e-3
        {
            return None;
        }
        let expected: f64 = probabilities
            .iter()
            .enumerate()
            .map(|(index, p)| index as f64 * p)
            .sum();
        let level = self
            .level_cut_points
            .iter()
            .filter(|cut| expected >= **cut)
            .count();
        let bracket = brackets
            .iter()
            .position(|bracket| bracket.id == self.level_brackets[level])?;
        Some((expected, level, bracket))
    }
}
