# Password & Secrets Management Policy

**Owner:** AI (Andre Ibiza)
**Last Updated:** 2026-01-14
**Tool:** Bitwarden (open source)

---

## 1. Principles

- **Single source of truth:** All credentials live in Bitwarden
- **Zero plaintext:** Never store secrets in code, notes, or unencrypted files
- **Minimal exposure:** Secrets accessed via environment variables, never hardcoded
- **Rotation discipline:** API keys rotated quarterly or on suspected compromise

---

## 2. Credential Categories

| Category | Examples | Vault Folder |
|----------|----------|--------------|
| **API Keys** | OpenAI, Anthropic, Poke, GitHub tokens | `Developer/API Keys` |
| **Service Accounts** | AWS, GCP, Supabase | `Developer/Cloud` |
| **Personal** | Email, banking, social | `Personal` |
| **MCP Connections** | Poke, other MCP servers | `Developer/MCP` |
| **SSH Keys** | Server access, Git signing | `Developer/SSH` |

---

## 3. Developer Workflow

### Local Development

```bash
# Install tools
brew install bitwarden-cli direnv

# Login to Bitwarden
bw login

# Project .envrc (committed - no secrets)
export POKE_API_KEY=$(bw get password "Poke API Key")
export OPENAI_API_KEY=$(bw get password "OpenAI API Key")

# Allow direnv
direnv allow
```

### Adding New Secrets

1. Create entry in Bitwarden under appropriate folder
2. Add env var reference to project `.envrc`
3. Run `direnv allow`
4. Never commit actual values

---

## 4. Security Rules

### DO

- Use Bitwarden CLI for automation
- Enable 2FA on Bitwarden account
- Use unique passwords per service
- Lock vault after 5 minutes of inactivity
- Back up vault export (encrypted) quarterly

### DON'T

- Share credentials via Slack/email/text
- Store secrets in `.env` files (use `.envrc` + Bitwarden)
- Commit any file containing real credentials
- Reuse passwords across services
- Screenshot credentials

---

## 5. Incident Response

**If a key is compromised:**

1. Revoke immediately at source (API dashboard)
2. Generate new key
3. Update Bitwarden entry
4. Run `direnv reload` in affected projects
5. Audit access logs if available
6. Document incident

---

## 6. File Hygiene

### Always .gitignore

```
.env
.env.*
.envrc.local
*.pem
*.key
credentials.json
secrets.yaml
```

### Safe to commit

```
.envrc  # If using Bitwarden CLI references only
.mcp.json  # With ${ENV_VAR} references
```

---

## 7. Current Integrations

| Service | Env Variable | Status |
|---------|--------------|--------|
| Poke MCP | `POKE_API_KEY` | ✅ Configured |
| Factory.ai | `FACTORY_TOKEN` | ⏳ Pending |
| Claude API | `ANTHROPIC_API_KEY` | ⏳ Pending |

---

*"Your initials are AI - secure your keys like your identity depends on it."*
