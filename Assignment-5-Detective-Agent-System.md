# Assignment 5 — Building a Detective Agent System

**Due:** Apr 27 by 11:59pm  
**Points:** 10  
**Submitting:** File upload  

## Background

A gruesome murder occurred. The local police department has collected a massive amount of data, but the lead detective has disappeared, leaving behind only a fragmented data dump.

You have been provided with all archived files containing the town's data (all files are in the “File/Assignment 5”):

- Transcripts of police interviews with witnesses and suspects.
- Town records, including driver's licenses, gym check-ins, social media event attendance, and vehicle logs.

## Your Mission

You are not solving this case manually. Instead, you must utilize LLMs for parsing the evidence, formulating hypotheses, cross-referencing alibis, and deducing the truth. Note that this case has layers: you must find not only the direct murderer, but also the mastermind who hired them.

## Initial information — Your starting point

> A murder occurred!!!  
> Someone killed the guard! He took an arrow to the knee! Security footage shows that there were 2 witnesses. The first witness lives at the last house on "Northwestern Dr". The second witness, named Annabel, lives somewhere on "Franklin Ave".

## Grading Rubrics

In this assignment, you need to show your ability to:

| Criterion | Points | Week |
|-----------|--------|------|
| Prompt LLMs appropriately | 1 pt | Week 7 |
| Construct Chain-of-Thoughts | 2 pt | Week 7 |
| Analyze the output of LLMs | 2 pt | Week 7 |
| Utilize tools / Function calling via LLMs | 2 pt | Week 8 |
| Construct and utilize RAG | 2 pt | Week 9 |
| Find the correct murderer and the brain behind the murder | 1 pt | — |

### Two additional score points (optional)

- Design a multi-agent system to analyze the murder automatically — **2 pt**
- Implement a multi-agent system (will be covered in Week 10) — **2 pt**

## Hints

- **Implement some tools:** For example, you may build an RAG to answer questions about a single interview. Then, implement some functions calling for the gym-lookup tool.
- **Start simple:** Don't build a multi-agent system immediately. You may start by interacting with your agent to analyze the cases. (If you cannot build an automatic system for it, then finding out the murderer in interactive mode is also OK.)
- **Beware of hallucinations:** LLMs love to invent evidence when they are stuck. If an agent claims someone was at the gym, ensure its prompt forces it to cite the specific tool output that proves it.
- **Follow the breadcrumbs:** The case is designed as a chain. A clue in the scene report leads to a witness; the witness gives a partial license plate; the plate leads to a suspect; the suspect's interview leads to the mastermind. Your CoT prompt must encourage the agents to follow this chain iteratively. Good luck!

## What to submit

1. Your code for the whole system (RAG, tool use, CoT, …).
2. A running history of your system, which shows the name of the murderer and the mastermind of this murder (**PDF**).
3. A report briefly illustrating how you utilize each function mentioned in the grading rubric (**PDF**).
