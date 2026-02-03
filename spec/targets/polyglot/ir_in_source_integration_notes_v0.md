### Tier A IR-in-source integration notes for polyglot targets

This note connects:
- Tier A ingest (`spec/ingest/ir_in_source/*`)
with:
- the polyglot target family plan (`spec/targets/polyglot/*`).

#### Principle
Any target language source file MAY carry a Tier A reference/embedded IR record.
That makes the source file a valid STUNIR input without needing to parse the language.

#### Suggested usage patterns
1) Reference form (preferred):
- Put `STUNIR_IR_REF` + `STUNIR_IR_SHA256` in a comment at the top of the source.
- Keep IR bundle stored in `spec/ir_bundle/`.

2) Embedded form:
- Embed base64url for portability in single-file contexts.

#### Determinism
Tier A ingest determinism comes from:
- sha256-verified bytes
- canonical IR verification

No language toolchains are needed.
