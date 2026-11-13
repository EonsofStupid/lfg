//! Suggested models: each known model's fit for the bracket's role, blended
//! with how the user actually uses it.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::config::AssessmentConfig;
use super::knowledge::{Knowledge, KnownModel};
use super::round2;
use super::text::contains_word;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSuggestion {
    pub subscription_id: String,
    pub model: String,
    pub label: String,
    pub description: String,
    pub role: Option<String>,
    pub effort: Option<String>,
    pub usual_effort: Option<String>,
    /// `provider-list`, `session` or `published`.
    pub origin: String,
    pub fit: f64,
    pub habit: f64,
    pub score: f64,
    pub prompts_in_bracket: u64,
    pub why: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelResult {
    pub role: String,
    pub effort: String,
    pub suggested: Option<ModelSuggestion>,
    pub alternatives: Vec<ModelSuggestion>,
    pub usual_model: Option<String>,
    pub plain: String,
}

fn described_role(config: &AssessmentConfig, description: &str) -> Option<String> {
    let text = description.to_lowercase();
    config
        .role_rules
        .iter()
        .find(|rule| {
            rule.words
                .iter()
                .any(|word| contains_word(&text, word, &config.word_suffixes))
        })
        .map(|rule| rule.role.clone())
}

fn family_words<'a>(config: &AssessmentConfig, model: &'a str) -> Vec<&'a str> {
    model
        .split('-')
        .filter(|word| {
            word.chars().any(char::is_alphabetic)
                && !config.family_ignore_words.iter().any(|skip| skip == word)
        })
        .collect()
}

/// Everything the assessment can rank: what the host learned, completed from the
/// published line-up, with each model's role read from its own description or,
/// failing that, inherited from a sibling in the same family.
pub fn known_models(config: &AssessmentConfig, knowledge: &Knowledge) -> Vec<KnownModel> {
    let mut models = knowledge.models.clone();
    for published in &config.published_models {
        match models
            .iter_mut()
            .find(|m| m.subscription_id == published.subscription_id && m.model == published.model)
        {
            Some(known) => {
                if known.description.is_empty() {
                    known.description = published.description.clone();
                }
                if known.label.is_empty() || known.label == known.model {
                    known.label = published.label.clone();
                }
                if known.efforts.is_empty() {
                    known.efforts = published.efforts.clone();
                }
            }
            None => models.push(KnownModel {
                subscription_id: published.subscription_id.clone(),
                model: published.model.clone(),
                label: published.label.clone(),
                description: published.description.clone(),
                efforts: published.efforts.clone(),
                origin: "published".into(),
                available: true,
                ..KnownModel::default()
            }),
        }
    }
    let direct: Vec<Option<String>> = models
        .iter()
        .map(|model| described_role(config, &model.description))
        .collect();
    let roles: Vec<Option<String>> = models
        .iter()
        .enumerate()
        .map(|(index, model)| {
            direct[index].clone().or_else(|| {
                let family = family_words(config, &model.model);
                models
                    .iter()
                    .enumerate()
                    .filter(|(other, sibling)| {
                        *other != index
                            && sibling.subscription_id == model.subscription_id
                            && direct[*other].is_some()
                            && family_words(config, &sibling.model)
                                .iter()
                                .any(|word| family.contains(word))
                    })
                    .min_by_key(|(_, sibling)| sibling.priority.unwrap_or(u32::MAX))
                    .and_then(|(other, _)| direct[other].clone())
            })
        })
        .collect();
    for (model, role) in models.iter_mut().zip(roles) {
        model.role = role;
    }
    models
}

fn fit_for(config: &AssessmentConfig, model: &KnownModel, target_role: &str) -> f64 {
    let m = &config.matching;
    let target = config
        .roles
        .iter()
        .position(|role| role == target_role)
        .unwrap_or(0);
    let base = match &model.role {
        None => m.unknown_role_fit,
        Some(role) => match config.roles.iter().position(|known| known == role) {
            None => m.off_ladder_fit,
            Some(position) => {
                (1.0 - m.role_step_penalty * position.abs_diff(target) as f64).max(m.minimum_fit)
            }
        },
    };
    let description = model.description.to_lowercase();
    let older = config
        .older_words
        .iter()
        .any(|word| contains_word(&description, word, &config.word_suffixes));
    if older {
        base * m.older_fit_factor
    } else {
        base
    }
}

fn weighted_near(config: &AssessmentConfig, model: &KnownModel, bracket: usize) -> f64 {
    let adjacent = config.matching.adjacent_bracket_weight;
    model
        .usage
        .iter()
        .filter_map(|usage| {
            let position = config.brackets.iter().position(|b| b.id == usage.bracket)?;
            match position.abs_diff(bracket) {
                0 => Some(usage.weighted),
                1 => Some(usage.weighted * adjacent),
                _ => None,
            }
        })
        .sum()
}

fn usual_effort(model: &KnownModel, bracket_id: &str) -> Option<String> {
    model
        .usage
        .iter()
        .find(|usage| usage.bracket == bracket_id)
        .and_then(|usage| {
            usage
                .efforts
                .iter()
                .max_by_key(|(_, count)| **count)
                .map(|(effort, _)| effort.clone())
        })
}

/// The target effort if this model supports it, otherwise the nearest level
/// below it, otherwise the model's lowest level.
pub(super) fn effort_for(
    config: &AssessmentConfig,
    model: &KnownModel,
    target: &str,
) -> Option<String> {
    if model.efforts.is_empty() {
        return None;
    }
    let rank = |effort: &str| config.effort_order.iter().position(|e| e == effort);
    let target_rank = rank(target)?;
    model
        .efforts
        .iter()
        .filter_map(|effort| rank(effort).map(|position| (position, effort)))
        .filter(|(position, _)| *position <= target_rank)
        .max_by_key(|(position, _)| *position)
        .or_else(|| {
            model
                .efforts
                .iter()
                .filter_map(|effort| rank(effort).map(|position| (position, effort)))
                .min_by_key(|(position, _)| *position)
        })
        .map(|(_, effort)| effort.clone())
}

fn rank_models(
    config: &AssessmentConfig,
    models: &[KnownModel],
    usable: &BTreeSet<String>,
    bracket_index: usize,
) -> Vec<ModelSuggestion> {
    let bracket = &config.brackets[bracket_index];
    let candidates: Vec<&KnownModel> = models
        .iter()
        .filter(|model| model.available && usable.contains(&model.subscription_id))
        .collect();
    let total: f64 = candidates
        .iter()
        .map(|model| weighted_near(config, model, bracket_index))
        .sum();
    let bracket_total: u64 = candidates
        .iter()
        .map(|model| prompts_in(model, &bracket.id))
        .sum();
    let influence = config.matching.usage_influence;
    let mut ranked: Vec<(ModelSuggestion, Option<u32>)> = candidates
        .iter()
        .map(|model| {
            let fit = fit_for(config, model, &bracket.role);
            let habit = if total > 0.0 {
                weighted_near(config, model, bracket_index) / total
            } else {
                0.0
            };
            let score = (1.0 - influence) * fit + influence * habit;
            let usual = usual_effort(model, &bracket.id);
            let effort = if config.matching.effort_follows_habit {
                usual
                    .clone()
                    .or_else(|| effort_for(config, model, &bracket.effort))
            } else {
                effort_for(config, model, &bracket.effort)
            };
            let prompts = prompts_in(model, &bracket.id);
            let role_text = model.role.as_deref().unwrap_or("unclassified");
            let usage_text = if prompts > 0 {
                let at = usual
                    .as_deref()
                    .map(|e| format!(", usually at {e} effort"))
                    .unwrap_or_default();
                format!(
                    "You used it for {prompts} of {bracket_total} {} requests{at}.",
                    bracket.label.to_lowercase()
                )
            } else {
                format!(
                    "You haven't used it for {} requests yet.",
                    bracket.label.to_lowercase()
                )
            };
            let why = format!("A {role_text} model. {usage_text}");
            (
                ModelSuggestion {
                    subscription_id: model.subscription_id.clone(),
                    model: model.model.clone(),
                    label: if model.label.is_empty() {
                        model.model.clone()
                    } else {
                        model.label.clone()
                    },
                    description: model.description.clone(),
                    role: model.role.clone(),
                    effort,
                    usual_effort: usual,
                    origin: model.origin.clone(),
                    fit: round2(fit),
                    habit: round2(habit),
                    score: round2(score),
                    prompts_in_bracket: prompts,
                    why,
                },
                model.priority,
            )
        })
        .collect();
    ranked.sort_by(|(a, a_priority), (b, b_priority)| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| {
                a_priority
                    .unwrap_or(u32::MAX)
                    .cmp(&b_priority.unwrap_or(u32::MAX))
            })
            .then_with(|| a.model.cmp(&b.model))
    });
    ranked.into_iter().map(|(model, _)| model).collect()
}

fn prompts_in(model: &KnownModel, bracket_id: &str) -> u64 {
    model
        .usage
        .iter()
        .find(|usage| usage.bracket == bracket_id)
        .map_or(0, |usage| usage.prompts)
}

/// Every model the host knows about with its role resolved, for inspection.
pub fn with_roles(config: &AssessmentConfig, knowledge: &Knowledge) -> Knowledge {
    Knowledge {
        models: known_models(config, knowledge),
        ..knowledge.clone()
    }
}

/// The suggested model for the chosen bracket among usable subscriptions, up to
/// the configured number of alternatives, and the model the user usually picks
/// for this bracket. Also returns how many models are known.
pub(super) fn model_result(
    config: &AssessmentConfig,
    knowledge: &Knowledge,
    usable: &BTreeSet<String>,
    chosen: usize,
) -> (ModelResult, usize) {
    let bracket = &config.brackets[chosen];
    let models = known_models(config, knowledge);
    let mut ranked = rank_models(config, &models, usable, chosen);
    let usual_model = ranked
        .iter()
        .filter(|model| model.prompts_in_bracket > 0)
        .max_by_key(|model| model.prompts_in_bracket)
        .map(|model| model.label.clone());
    let suggested = (!ranked.is_empty()).then(|| ranked.remove(0));
    ranked.truncate(config.matching.max_alternatives);
    let plain = match &suggested {
        None if usable.is_empty() => {
            "LFG found no subscriptions yet. Mark one below to get a suggestion.".to_string()
        }
        None => "LFG knows no models for your subscriptions yet.".to_string(),
        Some(model) => {
            let mut text = format!(
                "{} work suits a {} model at {} effort. ",
                bracket.label, bracket.role, bracket.effort
            );
            if model.role.as_deref() == Some(bracket.role.as_str()) {
                text.push_str(&format!("{} is the best match.", model.label));
            } else if model.prompts_in_bracket > 0 {
                text.push_str(&format!(
                    "{} ranks first because it is close to that and you use it most for this work.",
                    model.label
                ));
            } else {
                text.push_str(&format!("{} is the closest match you have.", model.label));
            }
            if let Some(usual) = usual_model.as_ref().filter(|usual| **usual != model.label) {
                text.push_str(&format!(" You usually use {usual} for this kind of work."));
            }
            text
        }
    };
    (
        ModelResult {
            role: bracket.role.clone(),
            effort: bracket.effort.clone(),
            suggested,
            alternatives: ranked,
            usual_model,
            plain,
        },
        models.len(),
    )
}
