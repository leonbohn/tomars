#![warn(unreachable_pub)]
#![warn(missing_docs)]
//! Tokenizer loading, byte-trie, token masks, and the mask cache (Phase 0.2-1.6).

pub(crate) mod tokens;
pub use tokens::{TokenId, TokenizerLoadError, Vocabulary};

mod trie;
