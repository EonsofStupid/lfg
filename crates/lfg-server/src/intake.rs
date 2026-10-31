use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Template {
    schema_version: String,
    revision: String,
    blocked_keys: Vec<String>,
    blocked_value_prefixes: Vec<String>,
    questions: Vec<Question>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Question {
    id: String,
    question: String,
    source_path: String,
    kind: ValueKind,
    required: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum ValueKind {
    String,
    Array,
    Object,
    Boolean,
    Number,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AppSnapshot {
    schema_version: String,
    source_app: String,
    facts: Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Preview {
    schema_version: &'static str,
    template_revision: String,
    template_digest: String,
    source_app: String,
    answers: Vec<Answer>,
    missing_required: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Answer {
    id: String,
    question: String,
    required: bool,
    status: &'static str,
    value: Option<Value>,
}

impl Template {
    pub fn parse(text: &str) -> Result<Self, String> {
        let template: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        let mut ids = BTreeSet::new();
        if template.schema_version != "lfg.intake-template.v1"
            || template.revision.trim().is_empty()
            || template.questions.is_empty()
            || template.blocked_keys.iter().any(|s| s.trim().is_empty())
            || template.blocked_value_prefixes.iter().any(|s| s.is_empty())
            || template.questions.iter().any(|question| {
                question.id.trim().is_empty()
                    || !ids.insert(&question.id)
                    || question.question.trim().is_empty()
                    || !valid_pointer(&question.source_path)
            })
        {
            return Err("invalid intake template".into());
        }
        Ok(template)
    }

    pub fn bundled() -> Self {
        Self::parse(include_str!("../config/intake-template.v1.json"))
            .expect("bundled intake template is valid")
    }

    pub fn digest(&self) -> String {
        lfg_core::digest_value(&serde_json::to_value(self).expect("intake template serializes"))
    }

    fn safe(&self, value: &Value) -> bool {
        match value {
            Value::Object(object) => object.iter().all(|(key, nested)| {
                !self
                    .blocked_keys
                    .iter()
                    .any(|blocked| blocked.eq_ignore_ascii_case(key))
                    && self.safe(nested)
            }),
            Value::Array(items) => items.iter().all(|item| self.safe(item)),
            Value::String(text) => !self
                .blocked_value_prefixes
                .iter()
                .any(|prefix| text.trim_start().starts_with(prefix)),
            _ => true,
        }
    }

    pub fn preview(&self, snapshot: AppSnapshot) -> Result<Preview, String> {
        if snapshot.schema_version != "lfg.app-snapshot.v1"
            || snapshot.source_app.trim().is_empty()
            || !snapshot.facts.is_object()
        {
            return Err("invalid app snapshot".into());
        }
        if !self.safe(&snapshot.facts) || !self.safe(&Value::String(snapshot.source_app.clone())) {
            return Err("app snapshot contains blocked credential-shaped data".into());
        }
        let mut missing_required = 0;
        let mut answers = Vec::new();
        for question in &self.questions {
            let selected = snapshot
                .facts
                .pointer(&question.source_path)
                .filter(|value| !value.is_null());
            let valid = selected.is_some_and(|value| match question.kind {
                ValueKind::String => value.as_str().is_some_and(|value| !value.trim().is_empty()),
                ValueKind::Array => value.is_array(),
                ValueKind::Object => value.is_object(),
                ValueKind::Boolean => value.is_boolean(),
                ValueKind::Number => value.is_number(),
            });
            let status = if selected.is_none() {
                "missing"
            } else if !valid {
                "invalid"
            } else {
                "filled"
            };
            if valid && !self.safe(selected.expect("valid value is present")) {
                return Err("selected facts contain blocked credential-shaped data".into());
            }
            if question.required && !valid {
                missing_required += 1;
            }
            answers.push(Answer {
                id: question.id.clone(),
                question: question.question.clone(),
                required: question.required,
                status,
                value: if valid { selected.cloned() } else { None },
            });
        }
        Ok(Preview {
            schema_version: "lfg.intake-preview.v1",
            template_revision: self.revision.clone(),
            template_digest: self.digest(),
            source_app: snapshot.source_app,
            answers,
            missing_required,
        })
    }
}

fn valid_pointer(path: &str) -> bool {
    if !path.starts_with('/') {
        return false;
    }
    let mut chars = path.chars();
    while let Some(character) = chars.next() {
        if character == '~' && !matches!(chars.next(), Some('0' | '1')) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fills_configured_questions_and_does_not_invent_missing_values() {
        let template = Template::bundled();
        let snapshot: AppSnapshot = serde_json::from_value(json!({
            "schemaVersion":"lfg.app-snapshot.v1", "sourceApp":"test-app",
            "facts":{"project":{"id":"project-one"},"request":{"text":"Build the feature"},
                     "providers":[{"id":"test-provider","activated":false,"capabilities":["ReadSessions"]}]}
        })).unwrap();
        let result = template.preview(snapshot).unwrap();
        let answers = serde_json::to_value(result).unwrap();
        let rows = answers["answers"].as_array().unwrap();
        let field = |id: &str| rows.iter().find(|row| row["id"] == id).unwrap();
        assert_eq!(field("project")["value"], "project-one");
        assert_eq!(field("user_request")["value"], "Build the feature");
        assert_eq!(field("providers")["value"][0]["activated"], false);
        assert_eq!(field("model_capabilities")["status"], "missing");
        assert!(field("model_capabilities")["value"].is_null());
        assert_eq!(answers["missingRequired"], 2);
    }

    #[test]
    fn custom_questions_are_dynamic_and_reject_bad_inputs() {
        let config = json!({"schemaVersion":"lfg.intake-template.v1","revision":"custom",
                            "blockedKeys":["api_key"],"blockedValuePrefixes":["secret-"],
                            "questions":[{"id":"flag","question":"Available?","sourcePath":"/flag","kind":"boolean","required":true}]});
        let template = Template::parse(&config.to_string()).unwrap();
        let request: AppSnapshot = serde_json::from_value(
            json!({"schemaVersion":"lfg.app-snapshot.v1","sourceApp":"app","facts":{"flag":false}}),
        )
        .unwrap();
        let value = serde_json::to_value(template.preview(request).unwrap()).unwrap();
        assert_eq!(value["answers"][0]["value"], false);
        assert_eq!(value["missingRequired"], 0);
        let unsafe_request: AppSnapshot = serde_json::from_value(json!({"schemaVersion":"lfg.app-snapshot.v1","sourceApp":"app","facts":{"flag":false,"api_key":"hidden"}})).unwrap();
        assert!(template.preview(unsafe_request).is_err());
        let mut invalid = config;
        invalid["questions"][0]["sourcePath"] = json!("/broken~2pointer");
        assert!(Template::parse(&invalid.to_string()).is_err());
    }
}
