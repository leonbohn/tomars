use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use serde::Deserialize;
use serde_json::Value;

use thiserror::Error;

/// Represents the vocabulary of a tokenizer, where each token is represented
/// as a sequence of bytes.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    tokens: Vec<Vec<u8>>,
    eos: TokenId,
}

impl Vocabulary {
    /// Construct from an instance of [`std::io::Read`].
    pub fn from_reader<R: std::io::Read>(reader: R) -> Result<Self, TokenizerLoadError> {
        Tokenizer::from_reader(reader)?.into_vocabulary()
    }

    /// Construct from a tokenizer given as a file.
    pub fn load_from_file<P: AsRef<Path>>(file_path: P) -> Result<Self, TokenizerLoadError> {
        Tokenizer::load_from_file(file_path)?.into_vocabulary()
    }

    pub(super) fn from_parts(tokens: Vec<Vec<u8>>, eos: TokenId) -> Self {
        Self { tokens, eos }
    }

    /// Get the bytes associated with a given [`TokenId`].
    pub fn token_bytes(&self, id: TokenId) -> Option<&[u8]> {
        self.tokens.get(id.0 as usize).map(AsRef::as_ref)
    }

    /// Get the [`TokenId`] of the special end of output token.
    pub fn eos(&self) -> TokenId {
        self.eos
    }

    /// Returns the size of the vocabulary, which is the number of disinct
    /// tokens. Also counts the eos token.
    pub fn size(&self) -> usize {
        self.tokens.len() + 1
    }
}

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
#[cfg(test)]
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
pub(crate) struct Tokenizer {
    added_tokens: Vec<AddedToken>,
    normalizer: Option<Value>,
    pre_tokenizer: Option<Pretokenizer>,
    model: BytePairEncoding,
}

/// Ways in which the loading of a tokenizer file and the construction of a
/// [`Vocabulary`] can fail.
/// The erorr variants together with the associated error messages should
/// hopefully document themselves.
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum TokenizerLoadError {
    #[error("could not read token file")]
    FileReadError(#[from] std::io::Error),
    #[error("could noto deserialize token file")]
    DeserializeError(#[from] serde_json::Error),
    #[error("tokenizer must not use normalization, are you sure you are using GPT-2?")]
    NormalizedTokenizer,
    #[error("tokenizer does not use byte level encoding")]
    NonByteLevel,
    #[error("tokenizer file is malformed")]
    MalformedFile,
    #[error("vocabulary contains empty token with id {0}")]
    EmptyTokenInVocabulary(u32),
    #[error("invalid character in token {0}")]
    InvalidTokenCharacter(char),
    #[error("duplicate token id {0}")]
    DuplicateTokenId(u32),
    #[error("missing token")]
    MissingTokenId,
    #[error("token id out of range {0}")]
    OutOfRangeTokenId(u32),
    #[error("end of text token is missing")]
    EndOfTextTokenMissing,
    #[error("vocabulary does not contain all single bytes as tokens")]
    ByteMissingFromVocabulary,
}

impl Tokenizer {
    pub(crate) fn load_from_file<P: AsRef<Path>>(file_path: P) -> Result<Self, TokenizerLoadError> {
        let file = std::fs::File::open(file_path)?;
        let reader = std::io::BufReader::new(file);
        Self::from_reader(reader)
    }

    pub(crate) fn from_reader<R: std::io::Read>(reader: R) -> Result<Self, TokenizerLoadError> {
        let tokenizer: Tokenizer = serde_json::from_reader(reader)?;

        if tokenizer.normalizer.is_some() {
            return Err(TokenizerLoadError::NormalizedTokenizer);
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

    pub(crate) fn into_vocabulary(self) -> Result<Vocabulary, TokenizerLoadError> {
        let Some(eos) = self.added_tokens.into_iter().find_map(|added| {
            if added.special && added.content == END_OF_TEXT_TOKEN {
                Some(added.id)
            } else {
                None
            }
        }) else {
            return Err(TokenizerLoadError::EndOfTextTokenMissing);
        };

        let mut tokens: Vec<Option<Vec<u8>>> = std::iter::repeat_n(None, eos.0 as usize).collect();

        for (unicode, token_id) in self.model.vocab.into_iter() {
            let index = token_id.0;
            if index > eos.0 {
                return Err(TokenizerLoadError::OutOfRangeTokenId(index));
            }
            if index == eos.0 {
                continue;
            }

            let mut token = Vec::with_capacity(unicode.len());
            for chr in unicode.chars() {
                let Some(byte) = unicode_to_byte(chr) else {
                    return Err(TokenizerLoadError::InvalidTokenCharacter(chr));
                };
                token.push(byte);
            }
            if token.is_empty() {
                return Err(TokenizerLoadError::EmptyTokenInVocabulary(index));
            }

            if tokens[index as usize].is_some() {
                return Err(TokenizerLoadError::DuplicateTokenId(index));
            }

            tokens[index as usize] = Some(token);
        }

        let Some(tokens) = tokens.into_iter().collect::<Option<Vec<_>>>() else {
            return Err(TokenizerLoadError::MissingTokenId);
        };

        let single_byte_tokens = tokens
            .iter()
            .filter_map(|tok| if tok.len() == 1 { Some(tok[0]) } else { None })
            .collect::<HashSet<_>>();
        if single_byte_tokens.len() != 256 {
            return Err(TokenizerLoadError::ByteMissingFromVocabulary);
        }

        Ok(Vocabulary::from_parts(tokens, eos))
    }
}

/// Newtype wrapper around a token id, wraps an `u32`.
#[derive(Deserialize, Debug, Clone, Copy, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[serde(transparent)]
pub struct TokenId(pub(crate) u32);

#[derive(Deserialize, Debug)]
pub(crate) struct AddedToken {
    id: TokenId,
    content: String,
    special: bool,
}

#[derive(Deserialize, Debug)]
pub(crate) struct BytePairEncoding {
    vocab: HashMap<String, TokenId>,
}

#[allow(dead_code)]
impl BytePairEncoding {
    pub(crate) fn get_token_id(&self, token: &str) -> Option<TokenId> {
        self.vocab.get(token).map(ToOwned::to_owned)
    }
    pub(crate) fn id_iter(&self) -> impl Iterator<Item = TokenId> {
        self.vocab.values().copied()
    }
    pub(crate) fn token_iter(&self) -> impl Iterator<Item = &String> {
        self.vocab.keys()
    }
    pub(crate) fn contains_token(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&String, TokenId)> {
        self.vocab.iter().map(|(a, b)| (a, *b))
    }
}

#[derive(Deserialize, Debug)]
pub(crate) struct Pretokenizer {
    r#type: String,
}

impl Pretokenizer {
    pub(crate) fn r#type(&self) -> &str {
        &self.r#type
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{num::u8, prelude::*};
    use serde_json::{Map, Value, json};
    use std::collections::HashSet;

    const VOCAB_SIZE: u64 = 50257;
    const EOS_ID: u64 = VOCAB_SIZE - 1;

    proptest! {
        #[test]
        fn unicode_to_byte_reversible(byte in u8::ANY) {
            assert_eq!(Some(byte), unicode_to_byte(byte_to_unicode(byte)))
        }
    }

    #[test]
    fn unicode_to_byte_known_values() {
        assert_eq!(unicode_to_byte('\u{0120}'), Some(0x20));
        assert_eq!(unicode_to_byte('\u{010A}'), Some(0x0A));
        assert_eq!(unicode_to_byte('\u{AC}'), Some(0xAC));
        assert_eq!(unicode_to_byte('\u{0143}'), Some(0xAD));
        assert_eq!(unicode_to_byte(' '), None);
        assert_eq!(unicode_to_byte('\u{AD}'), None);
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

    // ---------------------------------------------------------------- fixtures

    /// Bytes -> the string that would stand in the tokenizer file.
    fn encoded(bytes: &[u8]) -> String {
        bytes.iter().map(|&b| byte_to_unicode(b)).collect()
    }

    /// Structurally valid, GPT-2-sized tokenizer file with a synthetic vocabulary:
    /// ids 0..=255 are the single-byte tokens (id == byte), ids 256..EOS_ID are
    /// two-byte tokens in lexicographic order, EOS (`<|endoftext|>`) has the last id
    /// and is only listed in `added_tokens`.
    fn valid_tokenizer() -> Value {
        let mut vocab = Map::new();
        for b in 0..=255u8 {
            vocab.insert(encoded(&[b]), json!(b));
        }
        let mut id = 256u64;
        'fill: for a in 0..=255u8 {
            for b in 0..=255u8 {
                if id == EOS_ID {
                    break 'fill;
                }
                vocab.insert(encoded(&[a, b]), json!(id));
                id += 1;
            }
        }
        json!({
            "added_tokens": [{
                "id": EOS_ID,
                "content": "<|endoftext|>",
                "special": true,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": true
            }],
            "normalizer": null,
            "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true },
            "model": { "vocab": vocab }
        })
    }

    fn vocab_mut(v: &mut Value) -> &mut Map<String, Value> {
        v["model"]["vocab"].as_object_mut().unwrap()
    }

    fn key_of(v: &Value, id: u64) -> String {
        v["model"]["vocab"]
            .as_object()
            .unwrap()
            .iter()
            .find(|(_, value)| value.as_u64() == Some(id))
            .map(|(key, _)| key.clone())
            .unwrap()
    }

    /// Swap the string of token `id` for `new_key`; size and id density stay intact,
    /// so only the property under test is violated.
    fn replace_token(v: &mut Value, id: u64, new_key: &str) {
        let old = key_of(v, id);
        let vocab = vocab_mut(v);
        vocab.remove(&old);
        vocab.insert(new_key.to_string(), json!(id));
    }

    fn load(v: &Value) -> Result<Vocabulary, TokenizerLoadError> {
        Tokenizer::from_reader(v.to_string().as_bytes())?.into_vocabulary()
    }

    fn load_mutated(f: impl FnOnce(&mut Value)) -> Result<Vocabulary, TokenizerLoadError> {
        let mut v = valid_tokenizer();
        f(&mut v);
        load(&v)
    }

    // ------------------------------------------------------------ positive tests

    #[test]
    fn accepts_valid_tokenizer() {
        let vocab = load(&valid_tokenizer()).unwrap();
        assert_eq!(vocab.size(), VOCAB_SIZE as usize);
        assert_eq!(vocab.eos(), TokenId(EOS_ID as u32));
    }

    #[test]
    fn single_byte_tokens_decode_to_their_byte() {
        let vocab = load(&valid_tokenizer()).unwrap();
        for b in 0..=255u8 {
            assert_eq!(
                vocab.token_bytes(TokenId(b as u32)),
                Some(&[b][..]),
                "byte {b:#04x}"
            );
        }
    }

    #[test]
    fn two_byte_tokens_decode_in_order() {
        let vocab = load(&valid_tokenizer()).unwrap();
        assert_eq!(vocab.token_bytes(TokenId(256)), Some(&[0u8, 0][..]));
        assert_eq!(vocab.token_bytes(TokenId(257)), Some(&[0u8, 1][..]));
        assert_eq!(vocab.token_bytes(TokenId(256 + 256)), Some(&[1u8, 0][..]));
    }

    #[test]
    fn eos_is_not_a_regular_token() {
        let vocab = load(&valid_tokenizer()).unwrap();
        assert_eq!(vocab.token_bytes(vocab.eos()), None);
        assert_eq!(vocab.token_bytes(TokenId(EOS_ID as u32 + 1)), None);
    }

    #[test]
    fn eos_listed_in_vocab_with_matching_id_is_accepted_and_skipped() {
        let mut v = valid_tokenizer();
        vocab_mut(&mut v).insert("<|endoftext|>".into(), json!(EOS_ID));
        let vocab = load(&v).unwrap();
        assert_eq!(vocab.token_bytes(vocab.eos()), None);
        assert_eq!(vocab.size(), VOCAB_SIZE as usize);
    }

    // ------------------------------------------------- negative tests: file form

    #[test]
    fn rejects_normalizer() {
        let r = load_mutated(|v| v["normalizer"] = json!({ "type": "NFC" }));
        assert!(matches!(r, Err(TokenizerLoadError::NormalizedTokenizer)));
    }

    #[test]
    fn rejects_non_byte_level_pre_tokenizer() {
        let r = load_mutated(|v| v["pre_tokenizer"] = json!({ "type": "Metaspace" }));
        assert!(matches!(r, Err(TokenizerLoadError::NonByteLevel)));
    }

    #[test]
    fn rejects_missing_pre_tokenizer() {
        let r = load_mutated(|v| v["pre_tokenizer"] = Value::Null);
        assert!(matches!(r, Err(TokenizerLoadError::NonByteLevel)));
    }

    #[test]
    fn rejects_missing_added_tokens() {
        // Depending on your implementation this is MalformedFile or a dedicated EOS error.
        let r = load_mutated(|v| v["added_tokens"] = json!([]));
        assert!(r.is_err());
    }

    #[test]
    fn rejects_garbage_json() {
        let r = Tokenizer::from_reader(&b"{ not json"[..]);
        assert!(matches!(r, Err(TokenizerLoadError::DeserializeError(_))));
    }

    // -------------------------------------------- negative tests: vocabulary

    #[test]
    fn rejects_empty_token() {
        let r = load_mutated(|v| replace_token(v, 300, ""));
        assert!(matches!(
            r,
            Err(TokenizerLoadError::EmptyTokenInVocabulary(300))
        ));
    }

    #[test]
    fn rejects_character_outside_byte_table() {
        // a plain space never occurs in a byte-level vocabulary (it is encoded as 'Ġ')
        let r = load_mutated(|v| replace_token(v, 300, " "));
        assert!(matches!(
            r,
            Err(TokenizerLoadError::InvalidTokenCharacter(_))
        ));
    }

    #[test]
    fn rejects_vocabulary_missing_a_single_byte_token() {
        // replace the token for byte 0x41 by an unused two-byte token
        let r = load_mutated(|v| replace_token(v, 0x41, &encoded(&[255, 255])));
        assert!(matches!(
            r,
            Err(TokenizerLoadError::ByteMissingFromVocabulary)
        ));
    }

    #[test]
    fn rejects_duplicate_id() {
        // token 300 gets id 301: 301 is now used twice, 300 is a gap, size unchanged
        let r = load_mutated(|v| {
            let key = key_of(v, 300);
            vocab_mut(v).insert(key, json!(301));
        });
        assert!(matches!(r, Err(TokenizerLoadError::DuplicateTokenId(301))));
    }

    #[test]
    fn rejects_gap_in_ids() {
        // should detect that token with id 300 is missing
        let r = load_mutated(|v| {
            let key = key_of(v, 300);
            vocab_mut(v).remove(&key);
        });
        assert!(matches!(r, Err(TokenizerLoadError::MissingTokenId)));
    }

    #[test]
    fn rejects_id_beyond_eos() {
        // EOS is expected to carry the largest id
        let r = load_mutated(|v| {
            vocab_mut(v).insert(encoded(&[255, 255]), json!(EOS_ID + 5));
        });
        assert!(r.is_err());
    }

    #[test]
    fn rejects_eos_in_vocab_with_different_id() {
        let r = load_mutated(|v| {
            vocab_mut(v).insert("<|endoftext|>".into(), json!(5));
        });
        assert!(r.is_err());
    }

    #[test]
    fn rejects_eos_that_is_not_the_last_id() {
        let r = load_mutated(|v| v["added_tokens"][0]["id"] = json!(100));
        assert!(r.is_err());
    }

    // ------------------------------------------------- real file (optional)

    fn real_vocabulary() -> Option<Vocabulary> {
        let Ok(path) = std::env::var("GPT2_TOKENIZER") else {
            eprintln!("GPT2_TOKENIZER not set, skipping");
            return None;
        };
        Some(
            Tokenizer::load_from_file(path)
                .unwrap()
                .into_vocabulary()
                .unwrap(),
        )
    }

    #[test]
    fn real_gpt2_known_tokens() {
        let Some(vocab) = real_vocabulary() else {
            return;
        };
        assert_eq!(vocab.size(), 50257);
        assert_eq!(vocab.eos(), TokenId(50256));
        // ids taken from tokenizer.json: "Ġpart":636,"fter":637,"Ġkn":638,"Ġbec":639,"Ġtime":640
        let expected: [(u32, &[u8]); 5] = [
            (636, b" part"),
            (637, b"fter"),
            (638, b" kn"),
            (639, b" bec"),
            (640, b" time"),
        ];
        for (id, bytes) in expected {
            assert_eq!(vocab.token_bytes(TokenId(id)), Some(bytes), "id {id}");
        }
    }

    #[test]
    fn real_gpt2_has_no_empty_tokens() {
        let Some(vocab) = real_vocabulary() else {
            return;
        };
        for id in 0..50256u32 {
            let bytes = vocab.token_bytes(TokenId(id)).expect("dense ids");
            assert!(!bytes.is_empty(), "id {id}");
        }
    }
}
