use clap::{Parser, Subcommand};

mod errors;
mod hash;
mod ir_v1;
mod jcs;
mod path_policy;
mod validate;
mod verify_emit;
mod verify_pack;

use errors::StunirError;

#[derive(Parser, Debug)]
#[command(name = "stunir-native", version, about = "STUNIR native stages: validate + verify (pack/receipts)")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Validate that an IR is schema-valid and already canonical (RFC 8785 / JCS).
    Validate {
        /// Path to IR JSON.
        ir_json: String,

        /// Allow a single trailing LF (\n) after canonical JSON.
        #[arg(long, default_value_t = false)]
        allow_trailing_lf: bool,
    },

    /// Verify pack integrity / receipts.
    Verify {
        #[command(subcommand)]
        kind: VerifyKind,
    },
}

#[derive(Subcommand, Debug)]
enum VerifyKind {
    /// Verify Profile-3-style pack (root_attestation.txt + pack_manifest.tsv + objects/sha256).
    Pack {
        /// Root directory of the pack.
        #[arg(long, default_value = ".")]
        root: String,

        /// Objects directory relative to root.
        #[arg(long, default_value = "objects/sha256")]
        objects_dir: String,

        /// Pack manifest path relative to root.
        #[arg(long, default_value = "pack_manifest.tsv")]
        pack_manifest: String,

        /// Root attestation path relative to root.
        #[arg(long, default_value = "root_attestation.txt")]
        root_attestation: String,

        /// Strictly require that the manifest contains exactly the set of regular files under root (excluding objects/sha256/** and pack_manifest.tsv).
        #[arg(long, default_value_t = false)]
        check_completeness: bool,

        /// Optional: base64 Ed25519 public key (32 bytes) to verify root_attestation.txt.sig if present.
        #[arg(long)]
        ed25519_pubkey_b64: Option<String>,
    },

    /// Verify an emission receipt JSON (stunir.emit.v1.json)
    Emit {
        /// Path to emission receipt JSON.
        receipt_json: String,

        /// Root directory for resolving output relpaths.
        #[arg(long, default_value = ".")]
        root: String,
    },
}

fn main() {
    let cli = Cli::parse();

    let result: Result<(), StunirError> = match cli.cmd {
        Command::Validate {
            ir_json,
            allow_trailing_lf,
        } => validate::cmd_validate(&ir_json, allow_trailing_lf),

        Command::Verify { kind } => match kind {
            VerifyKind::Pack {
                root,
                objects_dir,
                pack_manifest,
                root_attestation,
                check_completeness,
                ed25519_pubkey_b64,
            } => verify_pack::cmd_verify_pack(
                &root,
                &objects_dir,
                &pack_manifest,
                &root_attestation,
                check_completeness,
                ed25519_pubkey_b64.as_deref(),
            ),

            VerifyKind::Emit { receipt_json, root } => {
                verify_emit::cmd_verify_emit(&receipt_json, &root)
            }
        },
    };

    match result {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(e.exit_code());
        }
    }
}
