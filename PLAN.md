# Projekt: Automatenbasiertes Constrained Decoding für LLMs (Rust)

## Ziel

Ein in Rust geschriebener **Token-Masker**, der für ein Sprachmodell in jedem Decodierschritt berechnet, welche Tokens mit einer formalen Spezifikation (Regex, kontextfreie Grammatik, optional ω-reguläre Eigenschaft) vereinbar sind. Grundlage ist meine eigene Automaten-Crate (NFA/DFA, ω-Automaten, Erreichbarkeit/Co-Erreichbarkeit, Partition Refinement, Minimierung, volle und lazy Determinisierung, generische Alphabete inkl. BDD-basierter Ausdrücke über atomare Propositionen).

Das Projekt soll drei Dinge zeigen:
1. **Korrektheit:** Masken sind nachweislich korrekt (Orakel-Tests, Differential Testing gegen llguidance).
2. **Engineering:** Messbare Performance mit nachvollziehbaren Optimierungen (Benchmarks mit Zahlen).
3. **Theorie:** Ein sauberes Experiment zur Verteilungsverzerrung durch Constrained Decoding (Grammar-Aligned Decoding), exakt berechnet statt geschätzt.

Das Sprachmodell ist **nur Mittel zum Zweck**. Kein eigener Transformer als Showcase, kein generierter Modellcode. Die zugrunde liegenden Begriffe (Attention, KV-Cache, Tokenizer, Sampling) lerne ich selbst so, dass ich sie erklären kann.

## Rahmen

- **Start:** Mi, 21.10.2026
- **Zeitbudget:** ca. 6 h Fokuszeit pro Tag, davon ca. 1 h Lernen
- **Bewerbungen:** ab spätestens Mo, 2.11.2026 parallel, mit Repo und README v1 als Nachweis
- **Ziel Jobstart:** Anfang Januar 2027
- **Hardware:** RX 5700 XT (RDNA1, kein offizieller ROCm-Support), daher alles Modellbezogene auf CPU. Der Masker selbst braucht keine GPU.

## Leitregeln

1. **Orakel vor Optimierung.** Jede schnelle Implementierung wird gegen eine naive, offensichtlich korrekte geprüft.
2. **Messen vor Optimieren.** Keine Optimierung ohne vorherige Messung, wo die Zeit hingeht.
3. **Zeitlimit pro Phase.** Überzieht eine Phase um mehr als 50 %, wird der Umfang gekürzt, nicht die Frist verlängert.
4. **README wächst mit.** Nach jeder Phase wird das README aktualisiert, nicht erst am Ende.
5. **Analyse- und Ausführungsschicht trennen.** Die Automaten-Crate (generisch) übernimmt Konstruktion und Analyse. Der heiße Pfad läuft auf konkreten, flachen Datenstrukturen ohne Traits.

## Nicht-Ziele

- Eigener Regex-Compiler (Parsing über `regex-syntax`)
- Eigener Transformer oder Trainingsstack in Rust
- GPU-Integration, Batching im Inferenzserver
- Unterstützung vieler Tokenizer gleichzeitig (GPT-2 zuerst, Llama 3 als Skalierungstest)

---

## Phase 0: Setup (Tag 1, Mi 21.10.)

### 0.1 Repository
- Cargo-Workspace anlegen mit den Crates:
  - `tomars-core`: Tokenizer-Laden, Trie, Masken, Cache
  - `tomars-regex`: Regex → NFA über Bytes → DFA (Adapter zur Automaten-Crate)
  - `tomars-cfg`: später, kontextfreie Grammatiken
  - `tomars-bench`: Benchmarks
  - `tomars-python`: später, Python-Bindings für das Experiment
- Abhängigkeiten: eigene Automaten-Crate, `tokenizers` oder `serde_json` (für `tokenizer.json`), `regex-syntax`, `fixedbitset` oder eigenes Bitset, `proptest`, `criterion`.
- Nix-Flake mit devShell (Rust-Toolchain, `cargo-nextest`, Python mit `torch` CPU, `safetensors`, `maturin` für später).
- CI: `cargo test`, `cargo clippy`, `cargo fmt --check`.

### 0.2 Tokenizer-Vokabular laden
- `tokenizer.json` von GPT-2 herunterladen (50.257 Tokens).
- Vokabular-Strings in **rohe Bytes** übersetzen. GPT-2-BPE speichert Tokens als Unicode-Kodierung der Bytes (`bytes_to_unicode`). Die Abbildung muss invertiert werden.
- Spezial-Tokens (z. B. `<|endoftext|>`) getrennt erfassen, nicht in den Trie.
- Datenstruktur: `Vec<Vec<u8>>` indiziert nach Token-ID.

### 0.3 Byte-Trie (naiv)
- Knoten mit Kindern als `Vec<(u8, NodeId)>`, optional `token: Option<TokenId>`.
- Einfügen aller Tokens.

### 0.4 Statistik und Tests
- Test: Für jedes Token t gilt `trie.lookup(bytes(t)) == Some(t)`.
- Test: Bytefolgen sind eindeutig (kein Knoten trägt zwei Tokens).
- Ausgabe: Vokabulargröße, Anzahl Trie-Knoten, maximale Tokenlänge, Anzahl Tokens mit **unvollständigem UTF-8** am Ende.

**Fertig, wenn:** alle Tests grün, Statistik im README notiert.

---

## Phase 1: Regex-Masker, korrekt (Tage 2-5, Do 22.10. – So 25.10.)

### 1.1 Regex → NFA über Bytes (Tag 2)
- Regex mit `regex-syntax` in HIR parsen.
- HIR in einen NFA der eigenen Crate übersetzen (Thompson-artig: Konkatenation, Alternative, Stern, Wiederholungen).
- Zeichenklassen in UTF-8-Bytesequenzen zerlegen (`regex_syntax::utf8::Utf8Sequences`). Übergänge als Bytebereiche oder Bytemengen (Expression-Transitionen der Crate).
- Anker und Lookarounds zunächst ablehnen (Fehler statt stiller Fehlinterpretation).
- Test: Für kleine Regexe stimmt die Sprache des NFA mit `regex::bytes::Regex` (vollständiger Match) überein, geprüft mit `proptest` an zufälligen Bytefolgen.

### 1.2 Trimmen und Determinisieren (Tag 2-3)
- NFA auf co-erreichbare Zustände trimmen. **Begründung festhalten:** Danach ist jeder nicht-tote Zustand lebendig, d. h. es gibt eine Fortsetzung zu einem akzeptierenden Zustand.
- Volle Determinisierung, dann Minimierung.
- Toten Zustand explizit modellieren.

### 1.3 Naiver Masker als Orakel (Tag 3)
- Für Zustand q und jedes Token t: Bytes von t einzeln durch den DFA, Token erlaubt genau dann, wenn Zielzustand lebendig.
- EOS-Regel: EOS nur erlaubt, wenn q akzeptierend.
- Dokumentieren: Laufzeit O(Σ Tokenlängen) pro Maske.

### 1.4 Trie-Masker (Tag 4)
- DFS über den Trie, DFA-Zustand wird mitgeführt.
- Bei totem Zustand: gesamten Teilbaum überspringen.
- Bei Knoten mit Token: Bit in der Maske setzen.
- Zählen: besuchte Trie-Knoten pro Maske.

### 1.5 Korrektheit (Tag 5)
- `proptest`: zufällige Regexe aus einer eingeschränkten Grammatik, zufällige erreichbare DFA-Zustände, **Trie-Maske == Orakel-Maske**.
- Handverlesene Regressionstests: JSON-Zahl, Datum, Bezeichner, Unicode-Klassen (z. B. `\p{L}+`), Regex mit großen Wiederholungszählern.
- Sonderfall prüfen: Tokens, die mitten in einem UTF-8-Zeichen enden.

### 1.6 Maskencache (Tag 5)
- Masken pro DFA-Zustand lazy berechnen und als Bitset cachen.
- Speicher pro Maske dokumentieren (GPT-2: ca. 6,3 KB, 128k-Vokabular: 16 KB).

**Fertig, wenn:** mehrere tausend Zufallsfälle grün, Cache funktioniert, README-Abschnitt „Korrektheit“ geschrieben.

---

## Phase 2: Schnell machen, messen, veröffentlichen (Mo 26.10. – So 1.11.)

### 2.1 Baseline messen (Tag 1)
- `criterion`-Benchmarks: Orakel vs. Trie, je für restriktive (`[0-9]+`), mittlere (JSON-String) und permissive (`.*`) Regexe.
- Profilieren (`perf`, `cargo flamegraph`): Wie viel Zeit geht in `step`, Speicherzugriffe, Trie-Navigation?

### 2.2 Kompilierter DFA (Tag 2)
- Byteklassen berechnen: Für jedes der 256 Bytes Signatur über alle vorkommenden Übergangsprädikate. Bytes mit gleicher Signatur bilden eine Klasse.
- Flache Übergangstabelle `table[state * stride + class]`, `u32`-Zustände, Senke als fester Index.
- Test: Kompilierter DFA verhält sich auf Zufallseingaben identisch zum Ursprungs-DFA.

### 2.3 Flacher Trie (Tag 3)
- Trie in Arrays umbauen (z. B. Kinder zusammenhängend und sortiert, Knoten-IDs in DFS-Reihenfolge).
- Optional: Teilbaumgröße pro Knoten speichern, um tote Teilbäume per Sprung zu überspringen.
- Erneut messen und vergleichen.

### 2.4 Sonderfälle (Tag 3-4)
- „Alles erlaubt“-Zustände erkennen (Restsprache Σ*) und volle Maske ohne Durchlauf zurückgeben.
- Jump-Forward: Maske mit genau einem Token → Token erzwungen, kein Modellaufruf nötig. Dokumentieren, dass erzwungene Tokenisierung von der kanonischen abweichen kann (Token-Healing-Problem).

### 2.5 Vergleich mit llguidance (Tag 4-5)
- Gleiche Regex, gleicher Tokenizer, gleicher Zustand: Masken vergleichen (Differential Testing).
- Zeit pro Maske vergleichen. Abweichungen erklären, nicht verstecken.

### 2.6 Llama-3-Tokenizer (Tag 5)
- Zweiter Tokenizer (128k Tokens) als Skalierungstest.
- Unterschiede im Vokabularformat behandeln.

### 2.7 README v1 (Tag 6-7)
- Problemstellung, Architektur (Analyse- vs. Ausführungsschicht), Korrektheitsargument (Trimmen ⇒ Lebendigkeit), Benchmarktabelle, Limitationen, nächste Schritte.
- Repo öffentlich machen.

**Fertig, wenn:** Repo öffentlich mit README und Zahlen. **Ab hier Bewerbungen.**

---

## Phase 3: Kontextfreie Grammatiken (Mo 2.11. – So 15.11.)

### 3.1 Grammatik-Eingabe
- Eine konkrete Zielgrammatik: JSON (ohne Schema).
- Format: einfache EBNF/Lark-Teilmenge, eigener kleiner Parser.
- Lexer/Parser-Trennung: Terminale als Regexe (wiederverwendet aus Phase 1), Struktur als CFG.
- Unproduktive und unerreichbare Nichtterminale entfernen (Analogon zum Trimmen).

### 3.2 Naiver Parser-Masker
- Earley-Parser oder PDA über Terminale.
- Pro Token: Bytes durch Lexer-DFA, bei Terminalgrenzen Parser fortschreiben, Token erlaubt, wenn der Parserzustand danach nicht leer ist.
- Orakel: identische Prüfung ohne Trie, Token für Token.

### 3.3 Persistenter Keller
- Keller als Baum mit Elternzeigern (Push = neuer Knoten, Pop = Zeiger nach oben, Kopie = Zeiger kopieren).
- Damit wird DFS über den Trie mit Kellerzustand billig.

### 3.4 Messen: kontextabhängige Tokens
- Für jeden Parserzustand zählen, wie viele Tokens beim Prüfen den Keller unterhalb der Spitze lesen.
- Ergebnis als Tabelle ins README. Diese Zahl begründet (oder widerlegt) die nächste Optimierung.

### 3.5 Optional: Aufteilung à la XGrammar
- Kontextunabhängige Tokens pro Zustand vorberechnen und cachen.
- Nur die kontextabhängigen zur Laufzeit am Keller prüfen.
- Vergleich mit 3.2 und mit llguidance.

**Fertig, wenn:** JSON-Masker korrekt gegen Orakel, Messung 3.4 im README. Wenn knapp: Phase 1/2 schärfen statt halbfertige CFG zeigen.

---

## Phase 4: Experiment zur Verteilungsverzerrung (Mo 16.11. – So 29.11.)

### 4.1 Modell (Mittel zum Zweck)
- nanoGPT-artiges Modell in **PyTorch auf CPU**, 1-10M Parameter, vorhandene Referenzimplementierung (nanoGPT), nicht selbst geschrieben.
- Training auf einer **Spielsprache**, z. B. Strings über kleinem Alphabet mit bekannter Struktur, damit das Modell eine nichttriviale Verteilung lernt.
- Eigener, kleiner Tokenizer (wenige Dutzend Tokens, darunter Mehrzeichentokens), damit die Token-/Grammatik-Fehlanpassung im Experiment vorkommt.

### 4.2 Anbindung
- Masker über Python-Bindings (`pyo3`/`maturin`) in die Generierungsschleife einhängen: Logits → Maske → Renormalisierung → Sampling.
- Schnittstelle bleibt modellunabhängig (`apply_mask(state, logits)`).

### 4.3 Exakte Berechnung statt Sampling
- Zielsprache L endlich und klein wählen (z. B. alle Wörter einer Grammatik bis Länge n).
- Für jedes w ∈ L: exakte Modellwahrscheinlichkeit P(w) über alle Tokenisierungen bzw. die kanonische (Entscheidung dokumentieren).
- Korrekte Zielverteilung: P(w | L) = P(w) / Σ_{w'∈L} P(w').
- Verteilung des Maskensamplers exakt berechnen: Produkt der renormalisierten bedingten Wahrscheinlichkeiten entlang jeder Ableitung.
- Abweichung messen: KL-Divergenz, Total Variation, Beispiele mit größter Verzerrung.

### 4.4 Optional: ASAp nachbauen
- Algorithmus aus dem GAD-Paper (adaptive Korrektur über approximierte erwartete Zukünfte) implementieren.
- Zeigen, wie die Abweichung mit der Zahl der Samples sinkt.

**Fertig, wenn:** ein Plot/Tabelle zur Verzerrung im README, Methodik nachvollziehbar beschrieben.

---

## Phase 5 (optional): ω-reguläre Spezifikationen

### 5.1 Lebendigkeit für ω-Automaten
- Pro Zustand eines Büchi-/Paritätsautomaten vorab prüfen, ob seine Restsprache nichtleer ist (SCC-Analyse, akzeptierender erreichbarer Zyklus).
- Zustände mit leerer Restsprache entfernen.
- Laufzeit: Teilmengenverfolgung über NBA-Zustände, Präfix lebendig ⇔ Menge nichtleer. Keine Safra-Konstruktion, kein Komplement.

### 5.2 Semantik klar benennen
- Die Maske verhindert nur Präfixe ohne gültige Fortsetzung. Sie erzwingt **keine** Lebendigkeitseigenschaften („irgendwann X“). Das muss im README eindeutig stehen.

### 5.3 Anwendung über atomare Propositionen
- Labeling: Abbildung von Tokens bzw. Aktionen auf Belegungen der APs (BDD-basierte Übergänge der Crate).
- Spielbeispiel: Agentenaktionen als Token-Alphabet, Sicherheitseigenschaft als Spezifikation.
- Offene Frage dokumentieren: Wie entsteht `label` bei Aktionen, die erst nach mehreren Tokens feststehen?

---

## Lernspur (parallel, ca. 1 h/Tag)

Ziel: Jeden Begriff so erklären können, dass ich in einem Gespräch Rückfragen beantworten kann. Pro Thema eine kurze eigene Notiz im Repo (`notes/`).

| Woche | Thema | Quelle | Selbsttest |
|---|---|---|---|
| 1 | Autoregressive Sprachmodelle, Softmax, Sampling (Temperatur, Top-k, Top-p) | Karpathy, Zero to Hero, Lecture 2 | Warum ändert Temperatur die Entropie, nicht die Rangfolge? |
| 1-2 | Attention, Q/K/V, kausale Maske, Multi-Head, Positionskodierung | „Attention Is All You Need“, Karpathy Lecture 7, Illustrated Transformer | Warum ist Attention ohne Positionskodierung permutationsäquivariant? |
| 2 | KV-Cache | eigene Herleitung aus Lecture 7 | Welche Kosten spart der KV-Cache pro Schritt, welche nicht? |
| 2 | Tokenizer, BPE, Byte-Level | Karpathy Lecture 8, `minbpe` | Warum können Tokens mitten in UTF-8-Zeichen enden? |
| 3 | Constrained Decoding | Grammar-Aligned Decoding (Park et al. 2024), XGrammar, llguidance-Doku | Warum verzerrt Maskieren mit Renormalisierung die Verteilung? |
| 4 | Expressivität von Transformern | Strobl et al., TACL 2024 (Survey) | Welche oberen Schranken (TC⁰) gelten unter welchen Annahmen? |

Links:
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Karpathy, Neural Networks: Zero to Hero: https://github.com/karpathy/nn-zero-to-hero
- minbpe: https://github.com/karpathy/minbpe
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- Grammar-Aligned Decoding: https://arxiv.org/abs/2405.21047
- XGrammar: https://arxiv.org/abs/2411.15100
- llguidance: https://github.com/guidance-ai/llguidance
- JSONSchemaBench / MaskBench: https://arxiv.org/abs/2501.10868
- Strobl et al., Survey: https://arxiv.org/abs/2311.00208

---

## Bewerbungsspur (parallel ab 2.11.)

- Bis 1.11.: CV und GitHub-Profil aktualisieren, Projekt als Hauptreferenz verlinken.
- Liste mit 15-20 Zielfirmen in drei Kategorien:
  - Inferenz-Infrastruktur / Structured Output
  - Verifikation und automatisiertes Schließen mit LLMs
  - Forschungsnahe Rollen zu Formalen Sprachen × Transformer
- Suchbegriffe: constrained decoding, structured generation, inference engineer, neurosymbolic, autoformalization, automated reasoning, program synthesis, formal verification LLM, runtime verification agents.
- Feste wöchentliche Zahl an Bewerbungen, selbst festgelegt.
- Pro Bewerbung: ein Satz, wie das Projekt zur Stelle passt.

---

## Risiken und Gegenmaßnahmen

| Risiko | Gegenmaßnahme |
|---|---|
| Verzettelung in Optimierungen | Leitregeln 1-3, Phasen-Deadlines |
| CFG-Phase zu groß | Auf JSON beschränken, notfalls nur naive Version plus Messung |
| Langsamer als llguidance | Erwartbar. Erklären, wo die Zeit hingeht, statt zu verstecken |
| Tokenizer-Sonderfälle (UTF-8, Spezial-Tokens, SentencePiece) | Früh testen (Phase 0.4, 1.5) |
| Modell-/GPU-Probleme | Modell nur auf CPU, nur in Phase 4, Masker modellunabhängig |
| Zeitplan für Januarstart | Bewerbungen ab 2.11. unabhängig vom Projektstand |

## Meilensteine

| Datum | Meilenstein |
|---|---|
| 21.10. | Vokabular und Trie mit Tests |
| 25.10. | Korrekter Regex-Masker mit Cache |
| 1.11. | Öffentliches Repo, Benchmarks, README v1 |
| 2.11. | Start Bewerbungen |
| 15.11. | JSON-Masker, Messung kontextabhängiger Tokens |
| 29.11. | Verzerrungs-Experiment im README |
| danach | Optional ω-Spezifikationen, Polishing, Beitrag zu llguidance |
