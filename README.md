# tomars

Automaton-based constrained decoding in Rust.

At each decoding step, `tomars` computes a token mask: which vocabulary tokens keep the output inside a formal specification. Regular expressions come first, then context-free grammars. ω-regular specifications may follow.

## Goals

- Correct token masking for byte-level BPE vocabularies (GPT-2 first).
- Every fast path is property-tested against a naive oracle.
- Benchmarks, including a comparison with [llguidance](https://github.com/guidance-ai/llguidance).
- Python bindings for experiments on how masking distorts the model's output distribution.

## Layout

| Crate | Purpose |
|---|---|
| `tomars-core` | Tokenizer loading, `Vocabulary`, byte trie |
| `tomars-regex` | Regex → byte NFA → DFA (uses the separate `automata` crate) |
| `tomars-cfg` | Context-free grammars (planned) |
| `tomars-bench` | Benchmarks (planned) |
| `tomars-python` | pyo3 bindings (planned) |

## Current state

Phase 0 is done or nearly done:

- GPT-2 `tokenizer.json` loader with strict sanity checks and negative tests.
- Byte-level vocabulary, round-trip tested against the `tokenizers` crate.
- Uncompressed byte trie over the vocabulary, with invariant and property tests.

Not yet implemented: regex front end, DFA, trie masker, benchmarks, CFG, Python bindings.

## Development

The repo uses a Nix flake (`nix develop`). Tests that need the GPT-2 tokenizer read its path from `GPT2_TOKENIZER`. The flake sets it. They skip when it is unset, and panic if `CI` is set.

```
cargo nextest run
```

`tomars-core` depends on the `automata` crate by path (`/sam/Rust/automata`).

## License

See `LICENSE`.
