# KHUB-PIL Demo Video Script
### "Your AI Assistant Should Remember What You Taught It"

**Target length:** 4–5 minutes
**Target audience:** Anyone who uses an AI assistant daily — no technical background needed
**Tone:** Conversational, curious, quietly impressive (not hype)
**Format notes for the creator:**
- Each `[SCREEN]` block describes what to show — terminal, diagram, or slide
- Each `NARRATOR:` block is the spoken voiceover
- `[CAPTION]` suggests on-screen text overlaid on the footage
- `[BEAT]` means a deliberate one-second pause — let the idea land before continuing
- Actual terminal commands are shown exactly as typed; the creator runs them live or records in advance

---

## SECTION 1 — THE PROBLEM (0:00 – 0:50)

---

**[SCREEN]** Calm, minimal slide. White background, one sentence appearing word by word:

> *"You told your AI assistant the same thing — for the fifth time."*

---

**NARRATOR:**
You've been there. You tell your AI assistant how you like your files named. It does it perfectly. Next week, you ask again — and it has no idea what you're talking about. You explain it again. It helps. The week after that — same thing.

**[BEAT]**

It's not broken. It's just... blank. Every conversation starts from zero. Everything you've ever taught it is gone.

---

**[SCREEN]** Split screen. Left: chat window labeled "Monday" — user types "always name statements YYYY-MM-institution.pdf". Right: chat window labeled "Friday" — user types the same thing again. A small counter in the corner ticks up: × 1 … × 2 … × 3 …

---

**NARRATOR:**
The frustrating part isn't the time it takes. It's that the knowledge was there. You did the work of explaining it. And the system just... didn't keep it.

**[BEAT]**

What if it could?

---

## SECTION 2 — INTRODUCING KHUB-PIL (0:50 – 1:30)

---

**[SCREEN]** Clean slide. Logo or project name fades in. Subtitle: *"Persistable Interactive Learning"*

---

**NARRATOR:**
KHUB-PIL — PIL stands for Persistable Interactive Learning — is a layer that sits alongside your AI assistant and gives it something it has never had before: memory that grows.

---

**[SCREEN]** Simple animated diagram — three boxes connected by arrows:

```
 Your conversation
       │
       ▼
 ┌─────────────┐
 │  KHUB-PIL   │  ← watches, learns, distills
 └─────────────┘
       │
       ▼
 Knowledge store
 (your machine, your files)
       │
       ▼
 Next conversation
 (agent already knows)
```

---

**NARRATOR:**
It watches what you say. It identifies patterns. It distils your preferences, your naming conventions, your workflows — into compact, readable notes stored right on your own machine. And the next time you start a conversation, those notes are already there.

Not because you hit "remember this." Just because you used it.

**[BEAT]**

Let me show you.

---

## SECTION 3 — THE DEMO (1:30 – 3:45)

*[Note to creator: the agent REPL is started with `pnpm --filter @khub-ai/computer-assistant start` from the repo root. Each session below is a separate run, or you can show them in one continuous session with a short visual cut between them. The `[PIL learned:]` lines appear automatically in the output — do not skip them; they are the payoff moments.]*

---

### Scene 3a — Session 1: one-off task, no learning (1:30 – 1:55)

**[SCREEN]** Terminal window. Prompt appears. Narrator types (or recording plays):

```
> Download my January statement from Chase, rename it
  "2026-01-chase-checking.pdf", and put it in my Finance/Statements folder.
```

Agent replies:
```
Done. Statement saved to ~/Finance/Statements/2026-01-chase-checking.pdf.
```

*[No `[PIL learned:]` line appears.]*

---

**NARRATOR:**
Session one. The agent does the job. One file, one task. PIL watches — but it sees a single data point. One instance isn't a pattern. So nothing is recorded yet.

**[CAPTION]** *"One task ≠ a habit. PIL waits for evidence."*

---

### Scene 3b — Session 2: pattern detected, convention born (1:55 – 2:30)

**[SCREEN]** New terminal prompt:

```
> Download my February statement from Chase. Same as last time.
```

Agent replies:
```
Got it — renamed to "2026-02-chase-checking.pdf" and saved
to ~/Finance/Statements/.

I notice you follow a consistent naming pattern:
YYYY-MM-institution-account.pdf. Want me to remember this?
```

User types:
```
> Yes, always use that pattern.
```

Agent replies:
```
[PIL learned: 1 new (convention)]
```

---

**NARRATOR:**
Second session. The agent notices something: you did the same thing the same way. It proposes a convention — and you confirm it in one word. That's it. From now on, this naming rule is part of what the agent knows about you.

**[BEAT]**

**[CAPTION]** *"Confirmed in one word. Remembered forever."*

---

### Scene 3c — Session 3: two things learned in one reply (2:30 – 3:00)

**[SCREEN]** New prompt:

```
> Get my February Fidelity statement too.
```

Agent:
```
Saved to ~/Finance/Statements/2026-02-fidelity-brokerage.pdf.

You now download statements from two institutions. Want a
monthly checklist so we don't miss any?
```

User:
```
> Good idea. Add Amex too — I always forget that one.
```

Agent:
```
[PIL learned: 2 new (fact, procedure)]
```

---

**NARRATOR:**
One reply — "Good idea, add Amex too" — and PIL extracts two separate things: the fact of which banks you use, and the full step-by-step procedure for your monthly download. Written down, structured, ready.

**[BEAT]**

**[CAPTION]** *"Two lessons from one sentence."*

---

### Scene 3d — Session 4: script generated (3:00 – 3:45)

**[SCREEN]** New prompt:

```
> Create a Python script to automate my monthly statement downloads.
```

Agent:
```
[Action: compile-procedure → python]
Script saved to ~/.openclaw/programs/monthly-statements-download-procedure.py
```

*[Cut to a second terminal window showing:]*
```bash
$ cat ~/.openclaw/programs/monthly-statements-download-procedure.py
```

*[The script scrolls — clearly structured Python with INSTITUTIONS list, download_statement(), main()]*

*[Cut back to the agent REPL:]*
```
> It's March 2nd — time for statements.
```

Agent:
```
[Action: run-command → python ~/.openclaw/programs/monthly-statements-download-procedure.py]
Found a script for this — running it now.
```

---

**NARRATOR:**
The agent has now seen this procedure enough times that it generates a Python script — automatically, using everything it learned: your institutions, your naming convention, your folder. And the next time you say "time for statements," it doesn't walk through the checklist manually. It runs the script. One line from you. Done.

**[BEAT]**

**[CAPTION]** *"From conversation → to convention → to procedure → to script."*

---

## SECTION 4 — THE STRONG POINTS (3:45 – 4:30)

---

**[SCREEN]** Clean slides, one point at a time, appearing as the narrator speaks.

---

**NARRATOR:**
A few things worth noticing about what just happened.

**[SCREEN]** Point 1 appears:

> **You never said "remember this."**

**NARRATOR:**
You just worked normally. PIL watched, learned, and proposed — and you confirmed with a word. That's the whole teaching interface.

---

**[SCREEN]** Point 2 appears:

> **The recipe is always kept alongside the script.**

**NARRATOR:**
The Python script is a convenient shortcut. But the human-readable checklist is always kept too. Why? Because after the script runs, you — or the agent — can check each step against it to make sure everything actually happened. The recipe is the source of truth. The script is just a faster way to follow it.

---

**[SCREEN]** Point 3 appears:

> **Your knowledge stays on your machine.**

**NARRATOR:**
Everything PIL learns is stored as plain text files in a folder on your computer. No cloud sync. No vendor database. No one can read it but you. And if you change AI providers tomorrow, you take everything with you.

---

**[SCREEN]** Point 4 appears:

> **It works with any AI assistant.**

**NARRATOR:**
PIL is not tied to one model or one platform. It's a layer — you bring your own AI. Claude, GPT, whatever comes next. The knowledge you build up belongs to you, not to whoever made the model.

---

## SECTION 5 — CLOSE (4:30 – 5:00)

---

**[SCREEN]** Return to the opening image — the split screen of the user typing the same thing twice — but now a green checkmark appears over both windows, and a small note fades in below:

> *"Only explained once. Applied from here on."*

---

**NARRATOR:**
An AI assistant that forgets everything isn't truly helpful — it's just fast. KHUB-PIL is the memory layer that turns fast into actually useful.

**[BEAT]**

If you'd like to try it, the full source code is open. There's a link below. And if you want to dig into how it works — the benchmarks, the design decisions, the roadmap — it's all documented.

**[BEAT]**

Thanks for watching.

---

**[SCREEN]** End card with:
- Project name / logo
- GitHub link
- A single line: *"Your AI assistant should earn its knowledge. And keep it."*

---

## PRODUCTION NOTES FOR THE CREATOR

### What to run before recording

```bash
# Clone and install
git clone https://github.com/khub-ai/openclaw-knowledge-management
cd openclaw-knowledge-management
pnpm install

# Verify the tests all pass (no API key needed)
pnpm test
# Expected: 111 passed

# Set your API key (needed for the live demo only)
export ANTHROPIC_API_KEY=sk-ant-...

# Start the agent REPL
pnpm --filter @khub-ai/computer-assistant start
```

### Terminal appearance tips

- Use a large, readable font (minimum 18pt) — viewers watch on small screens
- A dark terminal theme (e.g. One Dark, Dracula) reads better on video than light themes
- The `[PIL learned:]` lines are the visual payoff — pause 1–2 seconds after each one appears before speaking the next line
- For Scene 3d, open a second terminal window side-by-side to show the script on disk — this makes the "it actually saved a file" moment concrete

### What NOT to skip

- The "no `[PIL learned:]`" moment in Session 1 — it's critical for the audience to understand that PIL is selective, not recording everything
- The recipe-alongside-script explanation in Section 4 — this is a differentiator that most memory systems don't have and it directly addresses a real concern ("what if the script breaks?")

### Possible cuts for a shorter version (90-second cut)

If a 90-second cut is needed for social media, keep:
- 0:00–0:20 (pain point)
- 1:55–2:30 (convention born — the most satisfying moment)
- 3:00–3:30 (script generated)
- 4:30–4:45 (close)

Drop Sections 2, 3a, 3c, and most of Section 4.

### Optional: show the artifact file directly

At any point after Session 2, you can open the artifact store in a text editor:

```bash
cat ~/.openclaw/knowledge/artifacts.jsonl | python -m json.tool
```

This shows a skeptical viewer exactly what is being stored — plain, readable JSON on their own machine. It's a strong trust moment.
