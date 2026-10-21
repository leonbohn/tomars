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
        if b < 0x21 || (b > 0x7E && b < 0xA1) || b == 0xAD {
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
fn byte_to_unicode(needle: u8) -> char {
    let index = UNICODE_TO_BYTE
        .iter()
        .enumerate()
        .find_map(|(index, maybe_byte)| {
            maybe_byte.and_then(|byte| if byte == needle { Some(index) } else { None })
        });
    char::from_u32(index.expect("a byte cannot be found in lookup table") as u32)
        .expect("conversion to char must be possible")
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use proptest::{num::u8, prelude::*};

    use super::{UNICODE_TO_BYTE, byte_to_unicode, unicode_to_byte};

    proptest! {
        #[test]
        fn unicode_to_byte_reversible(byte in u8::ANY) {
            assert_eq!(Some(byte), unicode_to_byte(byte_to_unicode(byte)))
        }
    }

    #[test]
    fn unicode_to_byte_individual_values() {
        assert_eq!(unicode_to_byte('Ġ'), Some(0x20));
        assert_eq!(unicode_to_byte('Ċ'), Some(0x0A));
        assert_eq!(unicode_to_byte('¬'), Some(0xAC));
        assert_eq!(unicode_to_byte('\u{0143}'), Some(0xAD));
    }

    #[test]
    fn unicode_to_byte_bijectivity() {
        let Some(found) = UNICODE_TO_BYTE
            .iter()
            .copied()
            .filter(Option::is_some)
            .collect::<Option<HashSet<_>>>()
        else {
            panic!("should never happen, we filtered out Nones");
        };
        assert_eq!(
            found,
            (0..=255).into_iter().collect::<HashSet<_>>(),
            "mapping encoded by table is not surjective"
        );

        // it suffices now to verify that precisely 256 entries are not None
        let non_vacant_entry_count = UNICODE_TO_BYTE
            .iter()
            .copied()
            .filter(Option::is_some)
            .count();
        assert_eq!(non_vacant_entry_count, 256, "mapping not injective");
    }
}
