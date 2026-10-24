# tomars — automaton-based constrained decoding (Rust)

Token masker for LLM decoding: for each decoding step, compute which vocabulary tokens keep the output inside a formal spec (regex → later CFG → optionally ω-regular). Built on León's own automata crate. Goal: a public, well-tested repo with benchmarks that serves as the main reference for job applications (inference infra / structured output / formal methods × LLMs).

## Working with León

- PhD in formal language theory (ω-regular languages, automata learning). Deep automata knowledge, newer to LLM internals (transformers, tokenizers, sampling). Explain ML/inference concepts in technical depth, skip automata basics.
- **He writes the core algorithms himself** (loader, trie, masker, automaton adapters) — he wants to understand and explain every line. Claude: review, explain, point out bugs with the reason, suggest approaches, write tests/boilerplate **when asked**. Don't silently rewrite his implementations.
- Be short, direct, critical. No praise unless earned. Facts must be correct — say when something is unverified.
- Chat language: German. Code, comments, commit messages, docs: English.

## Principles (non-negotiable)

1. **Oracle before optimization.** Every fast path is property-tested against a naive, obviously correct version.
2. **Measure before optimizing.** No optimization without a prior measurement.
3. **Phase time limits.** If a phase overruns by >50 %, cut scope, don't extend.
4. **README grows with each phase.**
5. **Separate analysis and execution layers.** The automata crate (generic) does construction/analysis; the hot path runs on concrete flat data (`u8`, `TokenId`, tables), no traits/monomorphization in inner loops.
6. **Test the tests.** For each check, a negative test that fails when the check is removed.

## Environment

- NixOS, flake with `rust-overlay` + `rust-toolchain.toml`, devShell has rust-analyzer, cargo-nextest, maturin, Python (torch CPU, safetensors).
- GPT-2 `tokenizer.json` fetched via `pkgs.fetchurl` (pinned HF revision + SRI hash), exposed as env var **`GPT2_TOKENIZER`**. Not in the repo (nothing to gitignore). Tests needing it skip with a message when unset, but **panic if `CI` is set**.
- Automata crate: path dependency at `/sam/Rust/automata` (has an unrelated `dead_code` warning).
- GPU: RX 5700 XT (RDNA1, gfx1010) — no official ROCm. Everything model-related runs on CPU. The masker needs no GPU.

## Layout

- Workspace `tomars`.
- `tomars-core`: module `tokens` (file loading, byte table, `Vocabulary`), `trie.rs`. Public API minimal: `Vocabulary`, `TokenId`, `TokenizerLoadError` (`#[non_exhaustive]`). Deserialization structs (`Tokenizer`/`BytePairEncoding` — should become private, ideally renamed `TokenizerFile`), byte tables, trie internals: `pub(crate)`. Start private, open only when another crate needs it.
- Planned: regex crate (Regex → byte NFA → DFA adapter), CFG crate, `tomars-bench` (criterion in `benches/` with `harness = false`, plus a binary for deterministic stats / llguidance comparison; heavy deps like llguidance/criterion stay out of core), Python bindings (pyo3/maturin) for Phase 4.

## Domain knowledge already established

**GPT-2 tokenizer.json** (HF `tokenizers` format): pipeline `added_tokens → normalizer (null) → pre_tokenizer (ByteLevel: GPT-2 regex split + byte→unicode) → model (BPE: vocab, merges) → post_processor (offsets only)`. Only `model.vocab` and `added_tokens` are parsed; serde ignores the rest. `model` has no `type` field in this file. `merges` only needed later for canonical tokenization.

**bytes_to_unicode**: bijection of 256 bytes to visible chars. Bytes `0x21..=0x7E`, `0xA1..=0xAC`, `0xAE..=0xFF` (188) map to themselves; the other 68 (`0x00..=0x20`, `0x7F..=0xA0`, `0xAD`) get code points 256..=323 in ascending byte order. So `0x20→U+0120 'Ġ'`, `0x0A→U+010A 'Ċ'`, `0xAD→U+0143`. Implemented as `const` table `[Option<u8>; 324]` indexed by code point (compile-time asserted 256 entries). Known-value tests exist.

**Vocabulary**: `Vec<Vec<u8>>` indexed by `TokenId`, regular tokens only. EOS `<|endoftext|>` = id 50256, the largest id, lies **outside** the vector → can never enter the trie. `size() = tokens.len() + 1` (= logits/mask length). Load checks: no normalizer, ByteLevel pre-tokenizer, EOS from `added_tokens` (if also in vocab: same id, skipped), dense ids, no duplicates, no empty token, every char in byte table, all 256 single-byte tokens present, size 50257. All with dedicated `TokenizerLoadError` variants and negative tests. `Vocabulary::from_parts` exists for tests (no GPT-2 size check).

**Tokenization caveat**: The masker allows *any* token sequence whose bytes are valid; the model was trained on *canonical* BPE tokenizations. Matters for jump-forward (forced strings must be canonically tokenized) and for Phase 4 (P(w) = sum over tokenizations vs. canonical only — decide & document).

**Trie**: uncompressed (one node per byte), arena `Vec<Node>`, `NodeId`, root = node 0, `Node { token: Option<TokenId>, children: Vec<(u8, NodeId)> }` sorted by byte. Root and many inner nodes carry no token; every leaf must. Built by sorting tokens lexicographically (`[u8]: Ord` is bytewise) and maintaining a stack truncated to `lcp(prev, bytes) + 1` (not to the token length — `abc` then `axy` is the classic bug). Trie stores no bytes per token; the vocabulary is kept alongside (needed to advance the automaton after the model picks a token). Interface for the masker: `root`, `children` (non-allocating iterator), `token`, `lookup`. Compressed/flat layout only after Phase 2 measurements.

## Status (2026-10-24)

Done: loader + all sanity checks + negative tests; byte table; vocabulary; roundtrip tests against the `tokenizers` crate (fixed strings + proptest, green); trie build via LCP stack.
Just written, verify: full trie test module (invariants incl. in-degree/reachability/reconstruction/prefix-count, lookups incl. negatives, handmade cases, proptest narrow/wide alphabets, order independence, corruption `should_panic` tests, GPT-2 test, `#[ignore]` stats printer).
Open in Phase 0: run `cargo test print_gpt2_trie_stats -- --ignored --nocapture` and paste the numbers as **golden values** (node count, nodes with/without token, max token length, UTF-8 truncated-at-end vs. invalid). Optional: Python reference table (`tiktoken` `decode_single_token_bytes`) hashed with SHA-256 as golden value.

## Plan

| Date | Step |
|---|---|
| Sun 25.10. | Golden values; **1.1** Regex → HIR (`regex-syntax`) → NFA over bytes in the automata crate (char classes via `regex_syntax::utf8::Utf8Sequences`; reject anchors/lookarounds), proptest vs `regex::bytes` full match |
| Mon 26.10. | **1.2** trim NFA to co-reachable states (then non-dead ⇔ live), determinize, minimize, explicit dead state. **1.3** naive oracle masker (per token, bytes through DFA; EOS only in accepting states) |
| Tue 27.10. | **1.4** trie masker: DFS carrying DFA state, prune subtree on dead state, set bit on token nodes |
| Wed 28.10. | **1.5** proptest trie mask == oracle mask (random regexes + reachable states), regressions (JSON number, identifiers, `\p{L}+`, large repeats, UTF-8 splits). **1.6** lazy per-state bitset cache |
| Thu–Fri | **2.1** criterion baseline + profiling; **2.2** compiled DFA: byte classes (signature of each byte over all edge predicates), flat `table[state*stride+class]`, u32 states |
| Sat–Sun | **2.3** flat trie, remeasure; **2.4** all-allowed states, jump-forward |
| ~30.10. | **2.5** differential test + timing vs llguidance (expect to be slower; explain why) |
| 31.10.–1.11. | README v1 (problem, architecture, correctness argument, benchmark table, limitations), repo public, CV |
| **2.11.** | **Start applications** (independent of project state) |
| 2.–15.11. | **Phase 3** CFG: JSON only, lexer (regex DFAs) + Earley/PDA, persistent stack (parent-pointer tree), naive first, **measure share of context-dependent tokens**, then optionally XGrammar-style split |
| 16.–29.11. | **Phase 4** distortion experiment: nanoGPT (existing impl, PyTorch CPU, 1–10M params) trained on a toy language with a small tokenizer incl. multi-char tokens; masker via pyo3; finite target language L → exact P(w), exact P(w\|L), exact masked-sampler distribution, KL/TV. Optional ASAp (GAD paper) |
| optional | **Phase 5** ω-specs: per-state nonemptiness (SCC, accepting reachable cycle), trim, subset tracking at runtime, no Safra. Mask only prevents dead prefixes, does **not** enforce liveness — state this explicitly. AP labels via BDDs |

Deferred: Llama 3 tokenizer (128k) as scaling test.

## Not doing

Own regex compiler, own transformer as showcase, training in Rust/Burn, GPU, multiple tokenizers at once, inference-server batching.

## References

Grammar-Aligned Decoding (Park et al., NeurIPS 2024, arXiv 2405.21047) · XGrammar (arXiv 2411.15100) · llguidance (github.com/guidance-ai/llguidance, Rust, Earley + regex-derivative lexer) · JSONSchemaBench/MaskBench (arXiv 2501.10868) · Strobl et al., "What Formal Languages Can Transformers Express?" (TACL 2024) · Karpathy, nn-zero-to-hero (lectures 2, 7, 8) · minbpe.
