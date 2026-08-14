# Base benchmark — moonshotai/Moonlight-16B-A3B-Instruct

Untuned Moonlight-16B-A3B-Instruct against frozen benchmark `c370f8d0f7da5fa1`. This is the baseline every fine-tune is measured against; it is committed so comparisons never re-measure a constant.

## Headline

- **Objective (valid categories): 0.6327 over 49 items** — the number to beat.
- Raw objective incl. known-invalid: 0.5339 over 59 (see caveat).
- Rubric (heuristic): 0.4 over 20.
- Unscored (judge / prose / RAG-abstain): 41.

## Per category

| category | n | mean | valid | note |
|---|--:|--:|:--:|---|
| behavior | 10 | 0.8 | yes |  |
| coding | 20 | 0.5 | yes |  |
| factuality | 10 | 0.0 | yes |  |
| instruction_following | 10 | 0.5 | yes |  |
| long_context_rag | 10 | 1.0 | yes |  |
| mathematics | 15 | 0.8 | yes |  |
| reasoning | 15 | None | yes |  |
| technical_knowledge | 10 | None | yes |  |
| tool_calling | 10 | 0.05 | NO | harness provides no tool schema in the prompt, so the model answers conversationally instead of emitting a tool call; measures the harness, not the model. Needs a v2 with tool definitions injected. |
| trading_research | 10 | None | yes |  |

## Caveats that change what to fine-tune

- **tool_calling ~0 is not a model weakness.** The harness sends no tool schema, so the model answers in prose instead of calling a tool. Fixing this is a benchmark v2 task (inject tool definitions), not a fine-tuning target. Excluded from the headline.
- **factuality 0/10 IS real.** The model invents nonexistent modules/flags/papers and states confidently wrong facts (e.g. 16B active params instead of 3B, self-attention instead of MLA, a fictional 'Kaggle K2' GPU). A legitimate, high-value fine-tuning target.

## Conditions (must match for any comparison)

- decode: greedy, max_new_tokens=512, temperature=0.0
- transformers 4.57.6, device Tesla T4
- cost: 143 min, 3.19 tok/s, peak 11.61 GB

