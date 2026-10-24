use crate::{TokenId, Vocabulary};

#[derive(Debug)]
pub(crate) struct Trie {
    nodes: Vec<Node>,
}

impl Trie {
    pub(crate) fn from_tokens<'a, I: IntoIterator<Item = (TokenId, &'a [u8])>>(tokens: I) -> Self {
        let mut buf = tokens.into_iter().collect::<Vec<_>>();
        buf.sort_unstable_by_key(|(_, b)| *b);

        let mut nodes = Vec::with_capacity(buf.len());
        nodes.push(Node::from_parts(None, Vec::default()));

        let mut stack = Vec::from_iter([0usize]);
        let mut prev: &[u8] = &[];

        for (id, bytes) in buf {
            let l = lcp(prev, bytes);
            stack.truncate(l + 1);
            for b in &bytes[l..] {
                let new_node_id = nodes.len();
                nodes.push(Node::from_parts(None, Vec::default()));
                let node = nodes
                    .get_mut(*stack.last().expect("stack must not be empty"))
                    .expect("node must be present");
                node.children.push((*b, new_node_id.into()));
                stack.push(new_node_id);
            }
            let last_node = nodes.get_mut(*stack.last().unwrap()).unwrap();
            last_node.token = Some(id);
            prev = bytes;
        }

        Self { nodes }
    }

    pub(crate) fn build(vocabulary: &Vocabulary) -> Self {
        Self::from_tokens(vocabulary.token_iter())
    }

    pub(crate) fn root(&self) -> &Node {
        &self.nodes[0]
    }

    pub(crate) fn children(&self, node_id: NodeId) -> Option<&[(u8, NodeId)]> {
        self.nodes
            .get(node_id.index())
            .map(|node| node.children.as_slice())
    }

    pub(crate) fn child_for(&self, node_id: NodeId, byte: u8) -> Option<NodeId> {
        self.children(node_id).and_then(|array| {
            array.iter().find_map(|(child_byte, child_id)| {
                if *child_byte == byte {
                    Some(*child_id)
                } else {
                    None
                }
            })
        })
    }

    pub(crate) fn token(&self, node_id: NodeId) -> Option<TokenId> {
        self.nodes.get(node_id.index()).and_then(|node| node.token)
    }

    pub(crate) fn lookup(&self, bytes: &[u8]) -> Option<TokenId> {
        let mut cur = 0;
        for b in bytes {
            let Some(successor_id) = self.child_for(cur.into(), *b) else {
                return None;
            };
            cur = successor_id.0
        }
        self.nodes[cur as usize].token
    }

    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NodeId(u32);

impl From<usize> for NodeId {
    fn from(value: usize) -> Self {
        Self(value as u32)
    }
}

impl From<u32> for NodeId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl NodeId {
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug)]
pub struct Node {
    token: Option<TokenId>,
    children: Vec<(u8, NodeId)>,
}

impl Node {
    pub(crate) fn from_parts(token: Option<TokenId>, children: Vec<(u8, NodeId)>) -> Self {
        Self { token, children }
    }
    pub(crate) fn children_ids(&self) -> impl Iterator<Item = NodeId> {
        self.children.iter().map(|(_, id)| *id)
    }
}

fn lcp(left: &[u8], right: &[u8]) -> usize {
    left.iter().zip(right).take_while(|(l, r)| l == r).count()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap, HashSet};

    use super::*;
    use proptest::prelude::*;

    // ------------------------------------------------------------------ helpers

    /// Vocabulary without the GPT-2 size checks; ids are the positions, EOS comes last.
    fn vocabulary_from_bytes(tokens: Vec<Vec<u8>>) -> Vocabulary {
        let eos = tokens.len().into();
        Vocabulary::from_parts(tokens, eos)
    }

    /// Tokens given in the encoded alphabet of the tokenizer file (e.g. "\u{0120}a" = " a").
    fn vocabulary_from_token_iter<'a, I: IntoIterator<Item = &'a str>>(iter: I) -> Vocabulary {
        vocabulary_from_bytes(
            iter.into_iter()
                .map(|token| {
                    token
                        .chars()
                        .map(|chr| {
                            crate::tokens::unicode_to_byte(chr)
                                .expect("character outside the byte table")
                        })
                        .collect()
                })
                .collect(),
        )
    }

    /// (id, bytes) of every regular token.
    fn tokens_of(vocab: &Vocabulary) -> Vec<(TokenId, Vec<u8>)> {
        (0..vocab.size() as u32)
            .filter_map(|i| {
                vocab
                    .token_bytes(TokenId(i))
                    .map(|b| (TokenId(i), b.to_vec()))
            })
            .collect()
    }

    fn all_bytes() -> Vec<u8> {
        (0..=255u8).collect()
    }

    /// Set of (path, token) pairs, independent of the arena layout.
    fn canonical(trie: &Trie) -> BTreeSet<(Vec<u8>, TokenId)> {
        let idx = |n: NodeId| -> usize { n.0 as usize }; // adjust to your NodeId
        let mut out = BTreeSet::new();
        let mut path = Vec::new();
        let mut stack = vec![(0usize, 0usize, 0u8)];
        while let Some((id, depth, byte)) = stack.pop() {
            path.truncate(depth.saturating_sub(1));
            if depth > 0 {
                path.push(byte);
            }
            let node = &trie.nodes[id];
            if let Some(token) = node.token {
                out.insert((path.clone(), token));
            }
            for &(b, child) in &node.children {
                stack.push((idx(child), depth + 1, b));
            }
        }
        out
    }

    // ---------------------------------------------------------------------- lcp

    proptest! {
        #[test]
        fn lcp_props(
            a in prop::collection::vec(any::<u8>(), 0..32),
            b in prop::collection::vec(any::<u8>(), 0..32),
        ) {
            let k = lcp(&a, &b);
            prop_assert!(k <= a.len().min(b.len()));
            prop_assert_eq!(lcp(&a, &b), lcp(&b, &a));
            prop_assert_eq!(lcp(&a, &a), a.len());
            prop_assert_eq!(lcp(&b, &b), b.len());
            prop_assert_eq!(&a[..k], &b[..k]);
        }
    }

    // --------------------------------------------------------- invariants checker

    fn check_invariants(trie: &Trie, vocab: &Vocabulary) {
        let tokens = tokens_of(vocab);
        let idx = |n: NodeId| -> usize { n.0 as usize }; // adjust to your NodeId; root = node 0
        let n = trie.nodes.len();

        assert!(
            tokens.iter().all(|(_, b)| !b.is_empty()),
            "vocabulary contains an empty token"
        );
        assert!(trie.root().token.is_none(), "root has no associated token");

        // local structure and in-degrees
        let mut indegree = vec![0usize; n];
        for (id, node) in trie.nodes.iter().enumerate() {
            let bytes: Vec<u8> = node.children.iter().map(|(b, _)| *b).collect();
            assert!(
                bytes.windows(2).all(|w| w[0] < w[1]),
                "node {id}: child bytes not strictly ascending (unsorted or duplicate label)"
            );
            for &(_, child) in &node.children {
                let c = idx(child);
                assert!(c < n, "node {id}: child id {c} out of range ({n} nodes)");
                indegree[c] += 1;
            }
            if node.children.is_empty() && id != 0 {
                assert!(
                    node.token.is_some(),
                    "leaf node {id} has no associated token"
                );
            }
        }
        assert_eq!(indegree[0], 0, "root has a parent");
        for (id, &d) in indegree.iter().enumerate().skip(1) {
            assert_eq!(d, 1, "node {id} has {d} parents");
        }

        // reachability, path reconstruction, depth statistics
        let mut visited = vec![false; n];
        let mut found: HashMap<TokenId, Vec<u8>> = HashMap::new();
        let mut tokens_per_depth: Vec<usize> = Vec::new();
        let mut max_depth = 0usize;
        let mut path: Vec<u8> = Vec::new();
        let mut stack = vec![(0usize, 0usize, 0u8)]; // (node, depth, byte on the edge into it)
        while let Some((id, depth, byte)) = stack.pop() {
            assert!(
                !visited[id],
                "node {id} reached twice (cycle or shared child)"
            );
            visited[id] = true;

            path.truncate(depth.saturating_sub(1));
            if depth > 0 {
                path.push(byte);
            }
            max_depth = max_depth.max(depth);

            let node = &trie.nodes[id];
            if let Some(token) = node.token {
                if tokens_per_depth.len() <= depth {
                    tokens_per_depth.resize(depth + 1, 0);
                }
                tokens_per_depth[depth] += 1;
                assert!(
                    found.insert(token, path.clone()).is_none(),
                    "token {token:?} stored twice"
                );
            }
            for &(b, child) in &node.children {
                stack.push((idx(child), depth + 1, b));
            }
        }
        assert!(
            visited.iter().all(|&v| v),
            "some nodes are unreachable from the root"
        );

        // content equals the vocabulary
        let expected: HashMap<TokenId, &[u8]> =
            tokens.iter().map(|(id, b)| (*id, b.as_slice())).collect();
        assert_eq!(
            found.len(),
            expected.len(),
            "number of tokens stored in the trie"
        );
        for (id, bytes) in &found {
            assert_eq!(
                expected.get(id).copied(),
                Some(bytes.as_slice()),
                "token {id:?} is stored under the wrong path"
            );
        }

        // node count = root + one node per distinct non-empty prefix
        let prefixes: HashSet<&[u8]> = tokens
            .iter()
            .flat_map(|(_, b)| (1..=b.len()).map(move |k| &b[..k]))
            .collect();
        assert_eq!(n, 1 + prefixes.len(), "node count != 1 + distinct prefixes");

        // depth statistics
        let max_len = tokens.iter().map(|(_, b)| b.len()).max().unwrap_or(0);
        assert_eq!(max_depth, max_len, "max depth != longest token");
        let mut expected_per_depth = vec![0usize; max_len + 1];
        for (_, b) in &tokens {
            expected_per_depth[b.len()] += 1;
        }
        tokens_per_depth.resize(max_len + 1, 0);
        assert_eq!(tokens_per_depth, expected_per_depth, "tokens per depth");

        // root fan-out = number of distinct first bytes
        let first_bytes: HashSet<u8> = tokens.iter().map(|(_, b)| b[0]).collect();
        assert_eq!(
            trie.root().children.len(),
            first_bytes.len(),
            "root fan-out"
        );
    }

    /// Tokens are found, everything else is not. `extras` are the bytes tried as one-byte
    /// extensions of every token (all 256 for small vocabularies, a sample for GPT-2).
    fn check_lookups(trie: &Trie, vocab: &Vocabulary, extras: &[u8]) {
        let tokens = tokens_of(vocab);
        let token_set: HashSet<&[u8]> = tokens.iter().map(|(_, b)| b.as_slice()).collect();

        for (id, bytes) in &tokens {
            assert_eq!(
                trie.lookup(bytes.as_slice()),
                Some(*id),
                "lookup of {bytes:?}"
            );
        }
        assert_eq!(trie.lookup(&[]), None, "empty slice must not be a token");

        for (_, bytes) in &tokens {
            for k in 1..bytes.len() {
                let prefix = &bytes[..k];
                if !token_set.contains(prefix) {
                    assert_eq!(
                        trie.lookup(prefix),
                        None,
                        "prefix {prefix:?} is not a token"
                    );
                }
            }
            for &extra in extras {
                let mut longer = bytes.clone();
                longer.push(extra);
                if !token_set.contains(longer.as_slice()) {
                    assert_eq!(trie.lookup(&longer), None, "{longer:?} is not a token");
                }
            }
        }
    }

    fn check_all(vocab: &Vocabulary) {
        let trie = Trie::build(vocab);
        check_invariants(&trie, vocab);
        check_lookups(&trie, vocab, &all_bytes());
    }

    // ---------------------------------------------------------- handmade vocabularies

    #[test]
    fn trie_for_mini_vocabulary() {
        let vocab = vocabulary_from_token_iter(["a", "ab", "abc", "abd", "b", "ba"]);
        let trie = Trie::build(&vocab);
        check_invariants(&trie, &vocab);
    }

    #[test]
    fn handmade_vocabularies() {
        let cases: &[&[&str]] = &[
            &[],
            &["a"],
            &["a", "ab"],    // token is a prefix of another
            &["ab", "a"],    // reverse insertion order
            &["abc", "axy"], // same length, branch at depth 1
            &["abc", "abd"],
            &["a", "ab", "abc", "abd", "b", "ba"],
            &["a", "b", "c"],                          // only single bytes
            &["abcdef", "abcdxx", "abcxxx", "axxxxx"], // staggered branching
            &["\u{0120}a", "\u{0120}b", "\u{0120}"],   // leading encoded space
        ];
        for tokens in cases {
            check_all(&vocabulary_from_token_iter(tokens.iter().copied()));
        }
    }

    #[test]
    fn empty_vocabulary_gives_root_only() {
        let vocab = vocabulary_from_bytes(vec![]);
        let trie = Trie::build(&vocab);
        assert_eq!(trie.nodes.len(), 1);
        check_invariants(&trie, &vocab);
    }

    // ------------------------------------------------------- random vocabularies

    /// Few distinct bytes and short tokens: many prefix relationships.
    fn narrow_tokens() -> impl Strategy<Value = Vec<Vec<u8>>> {
        prop::collection::hash_set(prop::collection::vec(0u8..3, 1..6), 0..40)
            .prop_map(|set| set.into_iter().collect())
    }

    /// Whole byte range: wide root fan-out, few shared prefixes.
    fn wide_tokens() -> impl Strategy<Value = Vec<Vec<u8>>> {
        prop::collection::hash_set(prop::collection::vec(any::<u8>(), 1..4), 0..30)
            .prop_map(|set| set.into_iter().collect())
    }

    proptest! {
        #[test]
        fn random_narrow_vocabularies(tokens in narrow_tokens()) {
            check_all(&vocabulary_from_bytes(tokens));
        }

        #[test]
        fn random_wide_vocabularies(tokens in wide_tokens()) {
            check_all(&vocabulary_from_bytes(tokens));
        }

        /// Same token set in a different order: same trie content, same size.
        #[test]
        fn build_is_independent_of_vocabulary_order(tokens in narrow_tokens()) {
            let forward = vocabulary_from_bytes(tokens.clone());
            let mut reversed_tokens = tokens;
            reversed_tokens.reverse();
            let backward = vocabulary_from_bytes(reversed_tokens);

            let (t1, t2) = (Trie::build(&forward), Trie::build(&backward));
            prop_assert_eq!(t1.nodes.len(), t2.nodes.len());
            let bytes = |t: &Trie| -> BTreeSet<Vec<u8>> {
                canonical(t).into_iter().map(|(path, _)| path).collect()
            };
            prop_assert_eq!(bytes(&t1), bytes(&t2));
        }

        #[test]
        fn build_is_deterministic(tokens in narrow_tokens()) {
            let vocab = vocabulary_from_bytes(tokens);
            prop_assert_eq!(canonical(&Trie::build(&vocab)), canonical(&Trie::build(&vocab)));
        }
    }

    // ------------------------------------------------- tests of the checker itself

    /// Build the mini trie, corrupt it, run the checker.
    fn corrupted(f: impl FnOnce(&mut Trie)) {
        let vocab = vocabulary_from_token_iter(["a", "ab", "abc", "abd", "b", "ba"]);
        let mut trie = Trie::build(&vocab);
        f(&mut trie);
        check_invariants(&trie, &vocab);
    }

    #[test]
    #[should_panic(expected = "strictly ascending")]
    fn detects_unsorted_children() {
        corrupted(|t| t.nodes[0].children.reverse());
    }

    #[test]
    #[should_panic(expected = "stored twice")]
    fn detects_duplicate_token() {
        corrupted(|t| {
            for node in t.nodes.iter_mut().filter(|n| n.token.is_some()) {
                node.token = Some(TokenId(0));
            }
        });
    }

    #[test]
    #[should_panic(expected = "leaf")]
    fn detects_leaf_without_token() {
        corrupted(|t| {
            for node in t.nodes.iter_mut().filter(|n| n.children.is_empty()) {
                node.token = None;
            }
        });
    }

    #[test]
    #[should_panic(expected = "parents")]
    fn detects_orphaned_subtree() {
        corrupted(|t| {
            t.nodes[0].children.pop();
        });
    }

    #[test]
    #[should_panic(expected = "wrong path")]
    fn detects_swapped_tokens() {
        corrupted(|t| {
            let with_token: Vec<usize> = t
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.token.is_some())
                .map(|(i, _)| i)
                .collect();
            let (a, b) = (with_token[0], with_token[1]);
            let tmp = t.nodes[a].token;
            t.nodes[a].token = t.nodes[b].token;
            t.nodes[b].token = tmp;
        });
    }

    // ----------------------------------------------------------- real vocabulary

    fn gpt2_vocab() -> Option<Vocabulary> {
        let Ok(path) = std::env::var("GPT2_TOKENIZER") else {
            eprintln!("GPT2_TOKENIZER not set, skipping");
            return None;
        };
        Some(
            crate::tokens::Tokenizer::load_from_file(path)
                .unwrap()
                .into_vocabulary()
                .unwrap(),
        )
    }

    #[test]
    fn gpt2_trie_satisfies_invariants() {
        let Some(vocab) = gpt2_vocab() else { return };
        let trie = Trie::build(&vocab);
        check_invariants(&trie, &vocab);
        // sampled extensions only: 256 extensions x 50k tokens is slow in debug builds
        check_lookups(&trie, &vocab, &[0x00, 0x41, 0x80, 0xFF]);
        assert_eq!(
            trie.root().children.len(),
            256,
            "all single bytes are tokens"
        );
    }

    /// Run with `cargo test print_gpt2_trie_stats -- --ignored --nocapture`, then paste
    /// the numbers as golden values into a regular test.
    #[test]
    #[ignore = "prints statistics for the golden values"]
    fn print_gpt2_trie_stats() {
        let Some(vocab) = gpt2_vocab() else { return };
        let trie = Trie::build(&vocab);
        let tokens = tokens_of(&vocab);

        let with_token = trie.nodes.iter().filter(|n| n.token.is_some()).count();
        let (mut incomplete_end, mut invalid) = (0usize, 0usize);
        for (_, bytes) in &tokens {
            if let Err(e) = std::str::from_utf8(bytes) {
                // None: the bytes end in a truncated but so far valid sequence
                if e.error_len().is_none() {
                    incomplete_end += 1;
                } else {
                    invalid += 1;
                }
            }
        }
        eprintln!("tokens:               {}", tokens.len());
        eprintln!("nodes:                {}", trie.nodes.len());
        eprintln!("nodes with token:     {with_token}");
        eprintln!("nodes without token:  {}", trie.nodes.len() - with_token);
        eprintln!("root fan-out:         {}", trie.root().children.len());
        eprintln!(
            "longest token (bytes): {}",
            tokens.iter().map(|(_, b)| b.len()).max().unwrap_or(0)
        );
        eprintln!("utf-8 truncated at end: {incomplete_end}");
        eprintln!("utf-8 invalid elsewhere: {invalid}");
    }
}
