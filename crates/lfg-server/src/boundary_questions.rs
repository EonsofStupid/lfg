use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::intake;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Catalog {
    schema_version: String,
    revision: String,
    source_notice: String,
    #[serde(default)]
    artifact_paths: BTreeMap<String, String>,
    banks: Vec<Bank>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Bank {
    id: String,
    title: String,
    status: String,
    questions: Vec<Question>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Question {
    id: String,
    section: String,
    number: u32,
    question: String,
    source_term: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    domain: Option<String>,
    binding: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Attunement {
    schema_version: String,
    source_app: String,
    selections: Vec<Selection>,
    #[serde(default)]
    artifacts: BTreeMap<String, Value>,
    #[serde(default)]
    facts: Option<Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Selection {
    bank_id: String,
    #[serde(default)]
    sections: Option<Vec<String>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Preview {
    schema_version: &'static str,
    catalog_revision: String,
    catalog_digest: String,
    source_app: String,
    questions: Vec<PreparedQuestion>,
    missing_bindings: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PreparedQuestion {
    bank_id: String,
    question: Question,
    binding_status: &'static str,
    artifact: Option<Value>,
    artifact_source: Option<&'static str>,
    artifact_path: Option<String>,
}

impl Catalog {
    pub(crate) fn has_binding(&self, binding: &str) -> bool {
        self.banks
            .iter()
            .flat_map(|bank| &bank.questions)
            .any(|question| question.binding.as_deref() == Some(binding))
    }
    pub fn parse(text: &str) -> Result<Self, String> {
        let catalog: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        let mut bank_ids = BTreeSet::new();
        let mut question_ids = BTreeSet::new();
        if catalog.schema_version != "lfg.boundary-question-banks.v1"
            || catalog.revision.trim().is_empty()
            || catalog.source_notice.trim().is_empty()
            || catalog.banks.is_empty()
            || catalog.banks.iter().any(|bank| {
                bank.id.trim().is_empty()
                    || !bank_ids.insert(&bank.id)
                    || bank.title.trim().is_empty()
                    || bank.status.trim().is_empty()
                    || bank.questions.is_empty()
                    || bank.questions.iter().any(|question| {
                        question.id.trim().is_empty()
                            || !question_ids.insert(&question.id)
                            || question.section.trim().is_empty()
                            || question.number == 0
                            || question.question.trim().is_empty()
                            || question.source_term.trim().is_empty()
                            || question
                                .domain
                                .as_ref()
                                .is_some_and(|d| d.trim().is_empty())
                            || question
                                .binding
                                .as_ref()
                                .is_some_and(|b| b.trim().is_empty())
                    })
            })
        {
            return Err("invalid boundary question catalog".into());
        }
        if catalog.artifact_paths.iter().any(|(key, path)| {
            key.trim().is_empty()
                || !intake::valid_pointer(path)
                || !catalog.banks.iter().any(|bank| {
                    bank.questions
                        .iter()
                        .any(|question| question.binding.as_ref() == Some(key))
                })
        }) {
            return Err("invalid boundary artifact path".into());
        }
        Ok(catalog)
    }

    pub fn bundled() -> Self {
        Self::parse(include_str!("../config/boundary-question-banks.v1.json"))
            .expect("bundled boundary question catalog is valid")
    }

    pub fn digest(&self) -> String {
        lfg_core::digest_value(&serde_json::to_value(self).expect("catalog serializes"))
    }

    pub fn attune(
        &self,
        request: Attunement,
        intake: &intake::Template,
    ) -> Result<Preview, String> {
        if request.schema_version != "lfg.boundary-attunement.v1"
            || request.source_app.trim().is_empty()
            || request.selections.is_empty()
            || request.artifacts.keys().any(|key| key.trim().is_empty())
            || request
                .facts
                .as_ref()
                .is_some_and(|facts| !facts.is_object())
        {
            return Err("invalid boundary attunement".into());
        }
        let supplied =
            serde_json::to_value(&request.artifacts).map_err(|error| error.to_string())?;
        if !intake.safe(&supplied)
            || request
                .facts
                .as_ref()
                .is_some_and(|facts| !intake.safe(facts))
            || !intake.safe(&Value::String(request.source_app.clone()))
        {
            return Err("attunement contains blocked credential-shaped data".into());
        }
        let mut selected_banks = BTreeSet::new();
        let mut missing = BTreeSet::new();
        let mut questions = Vec::new();
        for selection in request.selections {
            if !selected_banks.insert(selection.bank_id.clone()) {
                return Err("duplicate selected bank".into());
            }
            let bank = self
                .banks
                .iter()
                .find(|bank| bank.id == selection.bank_id)
                .ok_or("unknown selected bank")?;
            let sections = if let Some(sections) = selection.sections {
                let count = sections.len();
                let unique: BTreeSet<_> = sections.into_iter().collect();
                if unique.is_empty()
                    || unique.len() != count
                    || unique
                        .iter()
                        .any(|s| !bank.questions.iter().any(|q| &q.section == s))
                {
                    return Err("unknown or empty section selection".into());
                }
                Some(unique)
            } else {
                None
            };
            for question in &bank.questions {
                if sections
                    .as_ref()
                    .is_some_and(|sections| !sections.contains(&question.section))
                {
                    continue;
                }
                let explicit = question
                    .binding
                    .as_ref()
                    .and_then(|key| request.artifacts.get(key));
                let path = question
                    .binding
                    .as_ref()
                    .and_then(|key| self.artifact_paths.get(key));
                let from_facts = || {
                    path.and_then(|path| {
                        request.facts.as_ref().and_then(|facts| facts.pointer(path))
                    })
                };
                let selected = explicit.or_else(from_facts);
                let artifact = selected.filter(|value| match value {
                    Value::Null => false,
                    Value::String(text) => !text.trim().is_empty(),
                    Value::Array(items) => !items.is_empty(),
                    Value::Object(object) => !object.is_empty(),
                    _ => true,
                });
                let binding_status = match (&question.binding, artifact) {
                    (None, _) => "not_required",
                    (Some(_), Some(_)) => "supplied",
                    (Some(key), None) => {
                        missing.insert(key.clone());
                        "missing"
                    }
                };
                questions.push(PreparedQuestion {
                    bank_id: bank.id.clone(),
                    question: question.clone(),
                    binding_status,
                    artifact: artifact.cloned(),
                    artifact_source: artifact.map(|_| {
                        if explicit.is_some() {
                            "app_artifact"
                        } else {
                            "app_fact"
                        }
                    }),
                    artifact_path: if artifact.is_some() && explicit.is_none() {
                        path.cloned()
                    } else {
                        None
                    },
                });
            }
        }
        Ok(Preview {
            schema_version: "lfg.boundary-attunement-preview.v1",
            catalog_revision: self.revision.clone(),
            catalog_digest: self.digest(),
            source_app: request.source_app,
            questions,
            missing_bindings: missing.into_iter().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn configured_artifact_paths_must_reference_a_real_binding_and_valid_pointer() {
        let mut value = serde_json::to_value(Catalog::bundled()).unwrap();
        value["artifactPaths"] = json!({"module inventory":"/project/modules"});
        assert!(Catalog::parse(&value.to_string()).is_ok());
        value["artifactPaths"] = json!({"module inventory":"/project/bad~2path"});
        assert!(Catalog::parse(&value.to_string()).is_err());
        value["artifactPaths"] = json!({"unknown artifact":"/project/modules"});
        assert!(Catalog::parse(&value.to_string()).is_err());
    }
}
