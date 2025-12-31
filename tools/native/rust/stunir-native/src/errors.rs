use thiserror::Error;

#[derive(Error, Debug)]
pub enum StunirError {
    #[error("IO: {0}")]
    Io(String),
    #[error("JSON: {0}")]
    Json(String),
    #[allow(dead_code)]
    #[error("Validation: {0}")]
    Validation(String),
    #[error("Verify Failed: {0}")]
    VerifyFailed(String),
}
