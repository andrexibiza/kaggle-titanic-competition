# Prompt Library & Knowledge Management Policy

**Owner:** AI (Andre Ibiza)
**Last Updated:** 2026-01-14
**Tools:** Bitwarden (secrets) | Poke (memory) | Git (versioned prompts)

---

## 1. The Problem

- Prompts scattered across chat histories, notes, random files
- No versioning - can't track what worked vs. what didn't
- Knowledge evaporates between sessions
- Reinventing the wheel on similar tasks

---

## 2. Principles

- **Prompts are code:** Version them, iterate them, test them
- **Single source of truth:** One canonical location for your prompt library
- **Build on success:** Tag what works, deprecate what doesn't
- **Context is king:** Pair prompts with their use cases and outcomes

---

## 3. Storage Architecture

```
~/.prompts/                     # Global prompt library
├── README.md                   # Index and usage guide
├── coding/
│   ├── code-review.md
│   ├── debug-error.md
│   ├── refactor.md
│   └── test-generation.md
├── writing/
│   ├── technical-docs.md
│   ├── email-professional.md
│   └── summarize.md
├── analysis/
│   ├── data-exploration.md
│   ├── root-cause.md
│   └── decision-matrix.md
├── personas/
│   ├── senior-engineer.md
│   ├── technical-writer.md
│   └── code-reviewer.md
└── templates/
    ├── PROMPT_TEMPLATE.md
    └── PERSONA_TEMPLATE.md
```

---

## 4. Prompt File Format

Each prompt file should follow this structure:

```markdown
# [Prompt Name]

## Metadata
- **Category:** coding | writing | analysis | persona
- **Version:** 1.2
- **Last Tested:** 2026-01-14
- **Success Rate:** high | medium | low
- **Best With:** Claude Opus | Sonnet | GPT-4 | etc.

## Use Case
When to use this prompt. What problem it solves.

## The Prompt
\`\`\`
[Your actual prompt here]

Variables to replace:
- {{CONTEXT}} - Description
- {{TASK}} - Description
- {{CONSTRAINTS}} - Description
\`\`\`

## Variables
| Variable | Description | Example |
|----------|-------------|---------|
| {{CONTEXT}} | Background info | "Python FastAPI codebase" |
| {{TASK}} | What you want done | "Review for security issues" |

## Examples

### Input
[Example of filled-in prompt]

### Output
[Example of good output received]

## Iteration Log
| Version | Date | Change | Result |
|---------|------|--------|--------|
| 1.0 | 2026-01-01 | Initial | Worked but verbose |
| 1.1 | 2026-01-07 | Added "be concise" | Better |
| 1.2 | 2026-01-14 | Added output format | Much better |

## Notes
- Works best with temperature 0.7
- Pair with senior-engineer persona for complex reviews
- See also: debug-error.md for follow-up
```

---

## 5. Prompt Categories

| Category | Purpose | Examples |
|----------|---------|----------|
| **Coding** | Development tasks | Review, debug, refactor, generate tests |
| **Writing** | Content creation | Docs, emails, summaries, explanations |
| **Analysis** | Thinking & decisions | Root cause, trade-offs, data exploration |
| **Personas** | Role-based context | Senior engineer, tech writer, critic |
| **Templates** | Starting points | New prompt scaffolding |
| **Chains** | Multi-step workflows | Research → Analyze → Recommend |

---

## 6. Integration with Your Stack

### Poke (Durable Memory)
```
Poke stores:
- Conversation context that persists
- Learned preferences over time
- Project-specific knowledge
- Successful prompt outcomes

Your prompt library stores:
- The prompts themselves (versioned)
- Templates and patterns
- Documentation
```

### Git Workflow
```bash
# Initialize prompt library
mkdir -p ~/.prompts/{coding,writing,analysis,personas,templates}
cd ~/.prompts
git init
git remote add origin git@github.com:yourusername/prompt-library.git

# Adding new prompt
cp templates/PROMPT_TEMPLATE.md coding/new-prompt.md
# Edit the file
git add . && git commit -m "Add new-prompt for X use case"
git push
```

### Quick Access (shell aliases)
```bash
# Add to ~/.zshrc
alias prompts="cd ~/.prompts && ls -la"
alias prompt="cat ~/.prompts"  # usage: prompt coding/code-review.md
alias pe="code ~/.prompts"     # open in editor

# Fuzzy find prompts (requires fzf)
fp() {
  local file=$(find ~/.prompts -name "*.md" | fzf --preview 'cat {}')
  [ -n "$file" ] && cat "$file"
}
```

---

## 7. Building Knowledge Over Time

### After Each Session
1. **Worked well?** → Save/update the prompt in library
2. **New pattern?** → Create new prompt file
3. **Failed?** → Note in iteration log, try variation
4. **Learned preference?** → Poke remembers, you document

### Weekly Review (15 min)
- [ ] Any prompts used 3+ times not in library? Add them.
- [ ] Any prompts that failed repeatedly? Deprecate or fix.
- [ ] Any new categories emerging? Create folder.
- [ ] Sync with Poke - is context building correctly?

### Monthly Audit
- [ ] Review iteration logs - what patterns emerge?
- [ ] Consolidate similar prompts
- [ ] Update success ratings
- [ ] Archive unused prompts (don't delete - move to `_archive/`)

---

## 8. Starter Prompts to Add

### Must-Have Prompts
1. **Code Review** - Your standard review checklist
2. **Debug Helper** - How you like errors analyzed
3. **Explain Code** - Your preferred explanation style
4. **Write Tests** - Your testing philosophy
5. **Refactor** - Your code quality standards
6. **Summarize** - How you like things condensed

### Your Personal Patterns
Document YOUR preferences:
- Do you like bullet points or prose?
- Verbose explanations or terse?
- Examples included or separate?
- What context do you always need to provide?

---

## 9. Anti-Patterns to Avoid

### DON'T
- Keep prompts only in chat history (they disappear)
- Use vague names like "prompt1.md" or "test.md"
- Skip the iteration log (you'll forget what you tried)
- Hoard prompts you never use (archive them)
- Forget to note which model works best

### DO
- Name prompts by function: `debug-python-error.md`
- Include at least one working example
- Track what model/settings worked
- Delete or archive ruthlessly
- Cross-reference related prompts

---

## 10. The Poke + Prompt Library Synergy

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR AI WORKFLOW                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Prompt Library (~/.prompts/)                          │
│   ├── Versioned templates                               │
│   ├── Documented patterns                               │
│   └── Iteration history                                 │
│              │                                          │
│              ▼                                          │
│   ┌─────────────────────┐                               │
│   │   Claude / Factory  │ ◄─── You provide prompt       │
│   └─────────────────────┘                               │
│              │                                          │
│              ▼                                          │
│   Poke (Durable Memory)                                 │
│   ├── Remembers what worked                             │
│   ├── Stores project context                            │
│   ├── Tracks your preferences                           │
│   └── Makes sessions coherent                           │
│                                                          │
└─────────────────────────────────────────────────────────┘

Prompts = the "what" (templates, patterns)
Poke = the "context" (memory, preferences, history)
Together = compounding knowledge
```

---

## 11. Quick Start Checklist

- [ ] Create `~/.prompts/` directory structure
- [ ] Initialize git repo for version control
- [ ] Add shell aliases for quick access
- [ ] Create your first 3 prompts from recent work
- [ ] Connect Poke for session memory
- [ ] Set calendar reminder for weekly review

---

*"Prompts are the new scripts. Version them like code, iterate like a scientist."*
