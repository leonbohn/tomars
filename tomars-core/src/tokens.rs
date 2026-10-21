use std::{collections::HashMap, path::Path};

use serde::Deserialize;
use serde_json::Value;

use thiserror::Error;

// We build a static lookup table for conversion of unicode to byte

static UNICODE_TO_BYTE: [Option<u8>; 324] = build_unicode_to_byte_table();

const fn build_unicode_to_byte_table() -> [Option<u8>; 324] {
    let mut tbl = [None; 324];
    let mut b = 0usize;
    let mut value = 256usize;

    while b < 256 {
        if b < 0x21 || (b > 0x7E && b < 0xA1) || b == 0xAC {
            tbl[value] = Some(b as u8);
            value += 1;
        } else {
            tbl[b] = Some(b as u8);
        }
        b += 1;
    }

    // sanity check that exactly 256 bytes have a place in encoding table
    let mut n = 0;
    let mut i = 0;
    while i < 324 {
        if tbl[i].is_some() {
            n += 1;
        }
        i += 1;
    }
    assert!(n == 256);

    tbl
}

fn unicode_to_byte(c: char) -> Option<u8> {
    UNICODE_TO_BYTE.get(c as usize).copied().flatten()
}

static END_OF_TEXT_TOKEN: &str = "<|endoftext|>";

#[derive(Deserialize, Debug)]
struct Tokenizer {
    added_tokens: Vec<AddedToken>,
    normalizer: Option<Value>,
    pre_tokenizer: Option<Pretokenizer>,
    model: BytePairEncoding,
}

#[derive(Debug, Error)]
pub enum TokenizerLoadError {
    #[error("could not read token file")]
    FileReadError(#[from] std::io::Error),
    #[error("could noto deserialize token file")]
    DeserializeError(#[from] serde_json::Error),
    #[error("tokenizer must not use deserialization, are you sure you are using GPT-2?")]
    NormalizedTokenizer,
    #[error("the endoftext token should not be present in the model itself")]
    EndOfTextInVocabulary,
    #[error("tokenizer does not use byte level encoding")]
    NonByteLevel,
    #[error("tokenizer file is malformed")]
    MalformedFile,
}

impl Tokenizer {
    pub fn load_from_file<P: AsRef<Path>>(file_path: P) -> Result<Self, TokenizerLoadError> {
        let file = std::fs::File::open(file_path)?;
        let reader = std::io::BufReader::new(file);
        let tokenizer: Tokenizer = serde_json::from_reader(reader)?;

        if tokenizer.normalizer.is_some() {
            return Err(TokenizerLoadError::NormalizedTokenizer);
        }

        if !tokenizer.model.contains_token(&END_OF_TEXT_TOKEN) {
            return Err(TokenizerLoadError::EndOfTextInVocabulary);
        }

        if tokenizer
            .pre_tokenizer
            .as_ref()
            .is_none_or(|pretokenizer| pretokenizer.r#type() != "ByteLevel")
        {
            return Err(TokenizerLoadError::NonByteLevel);
        }

        if tokenizer.model.vocab.is_empty() || tokenizer.added_tokens.is_empty() {
            return Err(TokenizerLoadError::MalformedFile);
        }

        Ok(tokenizer)
    }
}

#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(transparent)]
struct TokenId(pub(crate) u32);

#[derive(Deserialize, Debug)]
struct AddedToken {
    id: TokenId,
    content: String,
    special: bool,
}

#[derive(Deserialize, Debug)]
struct BytePairEncoding {
    vocab: HashMap<String, TokenId>,
}

impl BytePairEncoding {
    pub fn get_token_id(&self, token: &str) -> Option<TokenId> {
        self.vocab.get(token).map(ToOwned::to_owned)
    }
    pub fn contains_token(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }
}

#[derive(Deserialize, Debug)]
struct Pretokenizer {
    r#type: String,
}

impl Pretokenizer {
    pub fn r#type(&self) -> &str {
        &self.r#type
    }
}
