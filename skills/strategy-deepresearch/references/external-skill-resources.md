# External Skill Resources

This file records third-party resources that may be useful when evolving `strategy-deepresearch`. Treat them as inspiration, not trusted dependencies.

Source table reviewed:
- Notion: `Claude Code Learning Resources for Economics and Finance Researchers (Continuously Updated)`
- URL: `https://gen-li.notion.site/339195e07a238020b8aae6b5a1661f08?v=339195e07a2380c0ad01000c92c92011`
- Accessed via browser snapshot on 2026-04-29.

## Most Relevant Resources

| Resource | URL | Notion category/topic | Why it matters |
|---|---|---|---|
| ai-asset-pricing | `https://github.com/Alexander-M-Dickerson/ai-asset-pricing` | Application tools / Skills & MCP | Cross-agent empirical asset pricing workflow with `AGENTS.md`, `CLAUDE.md`, and reusable research scaffolding. Useful as a pattern for finance-specific agent onboarding, not as a direct dependency. |
| academic-research-skills | `https://github.com/franklee16/academic-research-skills` | Application tools | Claude Code skills for academic research in economics, finance, and social sciences. Relevant categories include literature review, data sourcing, peer review, visualization, and project management. |
| awesome-agent-skills | `https://github.com/heilcheng/awesome-agent-skills` | Application tools / Resource | Broad skill directory and quality guidance. Useful for benchmarking skill structure and discovery metadata. |
| avoid-ai-writing | `https://github.com/conorbronsdon/avoid-ai-writing` | Application tools / Skills & MCP | Example of a portable skill that explicitly targets Claude Code, OpenClaw, and Hermes compatibility. Useful for cross-agent packaging conventions. |
| How to create a Stata skill | `https://ai-mba.io/tutorials/how-to-create-a-stata-skill` | Learning materials | Domain-specific skill authoring example for economics workflows. Useful pattern for narrow procedural skills. |
| AI Agents for Economics Research | `https://ai-mba.io/tutorials/ai-agents-for-economics-research` | Learning materials / General | General guidance on agent workflows for economics research. Use for process comparison only. |
| claude-wrds-public | `https://github.com/lsy617004926/claude-wrds-public` and `https://github.com/piotrek-orlowski/claude-wrds-public` | Application tools / Research Application | WRDS-oriented research workflow examples. Useful if `strategy-deepresearch` later adds data provenance checks against finance datasets. |
| idea-evaluation-pipeline | `https://github.com/alejandroll10/idea-evaluation-pipeline` | Application tools | Research idea triage and evaluation workflow. Useful for inspiration on evidence matrices and review loops. |
| corbis-literature-starter-kit | `https://github.com/Agentic-Assets/corbis-literature-starter-kit` | Application tools | Literature workflow starter. Relevant if adding richer source collection/reporting. |

## Collection Notes

- The Notion page is a resource index, not a package registry. Do not assume every entry is a ready-to-install skill.
- Prefer extracting patterns into this local skill instead of adding dependencies.
- Do not install third-party skills into `~/.codex/skills` or `~/.hermes/skills` without separately reviewing their `SKILL.md`, scripts, licenses, and side effects.
- Finance-specific resources are more relevant than general writing or presentation skills for this audit workflow.

## Safe Reuse Patterns

- Keep skill instructions short and procedural.
- Separate long checklists and templates into `references/`.
- Add deterministic local context collection scripts where repeated manual inspection would be error-prone.
- Make cross-agent compatibility explicit in the skill body and test with the target agent CLI.
- Keep audit workflows sidecar-only when the main optimization loop must remain reproducible.
