# Security Policy

## Reporting a Vulnerability
Please disclose security issues privately first. Use one of the following channels:
- Open a [GitHub Security Advisory draft](https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories) targeting this repository.
- Or email the core maintainers (contact details in the repository README or organization profile) with the subject **"[Security] Agent Market"**.

Include:
- A concise description of the vulnerability and potential impact.
- Steps to reproduce or proof-of-concept.
- Suggested remediation ideas if available.

We will acknowledge reports within five business days and keep you informed about progress. Do not create a public issue until a fix is released.

## Supported Versions
- `main` (development) – receives security updates continuously.
- Release branches (e.g., `v0.1.x` once published) – critical fixes only.

## Secrets & Credentials
- Never commit real API keys or `.env` files. Use `.env.example` as a template.
- Local development should rely on ephemeral credentials. Production deployments must integrate with a secret manager (see TODO-12/13).
- Rotate keys after every incident or when sharing access with new collaborators.

## Dependency Updates
- Prefer deterministic upgrades via `requirements*.txt` and `package-lock.json`.
- Run `ruff check .`, `black --check .`, and `pytest -q` after upgrading.
- Review vulnerability alerts surfaced by `npm audit` and GitHub Dependabot.

## Secure Deployment Checklist
1. Serve the API behind TLS and apply network policies to restrict access.
2. Enable authentication (API keys or OAuth proxy) before exposing endpoints beyond a trusted network.
3. Store `resources/user_data/app.db` and artefacts on encrypted volumes with regular backups.
4. Configure monitoring (OpenTelemetry/Prometheus integration is tracked under TODO-42).
5. Keep Docker images and Conda environments patched; rebuild on a regular cadence.

Thank you for helping keep Agent Market secure.
