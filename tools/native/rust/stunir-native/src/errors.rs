use std::fmt;

#[derive(Debug)]
pub enum StunirError {
    Usage(String),
    VerifyFailed(&'static str),
    Io(&'static str),
    Utf8(&'static str),
    Json(&'static str),
}

impl StunirError {
    pub fn exit_code(&self) -> i32 {
        match self {
            StunirError::VerifyFailed(_) => 1,
            StunirError::Usage(_) => 2,
            StunirError::Io(_) | StunirError::Utf8(_) | StunirError::Json(_) => 1,
        }
    }
}

impl fmt::Display for StunirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StunirError::Usage(msg) => write!(f, "usage error: {}", msg),
            StunirError::VerifyFailed(tag) => write!(f, "{}", tag),
            StunirError::Io(tag) => write!(f, "{}", tag),
            StunirError::Utf8(tag) => write!(f, "{}", tag),
            StunirError::Json(tag) => write!(f, "{}", tag),
        }
    }
}

impl std::error::Error for StunirError {}
