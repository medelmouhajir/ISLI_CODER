# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

---

## Security Architecture & Tool Execution

ISLI operates as an AI coding assistant with terminal and filesystem capabilities. To safeguard developer environments, ISLI includes built-in security features:

- **Permission Safety Modes**:
  - `normal` (Default): All state-altering tools (`write`, `edit`, `bash`, `git`) require explicit confirmation by the user before execution.
  - `plan`: Read-only analysis mode. Blocks all write, edit, and shell operations while permitting read/grep/glob analysis and plan tracking.
  - `auto`: Uses local Keeper SLM to classify tool invocations as safe or risky before prompting.
  - `robot`: Autonomous mode for controlled CI environments; explicit deny rules remain strictly enforced.
- **Directory Traversal Protection**:
  - File access is strictly contained within the project workspace using path canonicalization (`resolve_path`). Attempts to escape outside the workspace root are rejected.
- **Process Cancellation & Timeouts**:
  - Shell commands and background tasks feature bounded execution timeouts and propagate graceful `SIGTERM` followed by `SIGKILL` / `taskkill` upon cancellation.

---

## Reporting a Vulnerability

We take the security of ISLI and our users seriously. If you discover a security vulnerability, please **do not open a public issue**.

Instead, please report vulnerabilities via:
1. **GitHub Private Vulnerability Reporting**: Submit a private advisory directly through [GitHub Security Advisories](https://github.com/medelmouhajir/ISLI_CODER/security/advisories/new).
2. **Direct Email**: Send encrypted details to Mohamed Amin EL-MOUHAJIR at **`med.elmouhajir@gmail.com`** with the subject `[SECURITY VULNERABILITY] ISLI`.

### What to Include
- A description of the vulnerability and its potential impact.
- Clear reproduction steps, proof of concept (PoC), or environment details.
- Any suggested remediations or mitigations.

### Response Timeline
- We will acknowledge receipt of your report within 48 hours.
- We will provide regular status updates while working on a patch.
- We will credit responsible disclosures in release notes once the patch is published.
