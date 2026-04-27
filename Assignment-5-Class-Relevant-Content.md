# Assignment 5 Class-Relevant Content

This note collects the course material in `AIforDesign_CourseFiles` that is most relevant to `Assignment-5-Detective-Agent-System.md`, especially the rubric items:

- prompt LLMs appropriately
- construct Chain-of-Thoughts (CoT)
- analyze LLM outputs
- utilize tools / function calling
- construct and utilize RAG
- optionally design a multi-agent system

## Source Files Reviewed

Most relevant notebooks:

- `AIforDesign_CourseFiles/Labs/Week7/05_prompt_engineering_tutorial_new.ipynb`
- `AIforDesign_CourseFiles/Labs/Week8/07_llm functional prototype.ipynb`
- `AIforDesign_CourseFiles/Labs/Week8/07_llm_funtion_call.ipynb`
- `AIforDesign_CourseFiles/Labs/Week9/09_RAGwithMongoDB.ipynb`
- `AIforDesign_CourseFiles/Labs/Week10/Lab09_MultiAgent_and_Human_AI_Cooperation.ipynb`
- `AIforDesign_CourseFiles/Labs/Week11/Lab10_Skills_and_MCP.ipynb`

Less relevant for this assignment:

- `Week6` Streamlit notebooks: UI setup, not core to the rubric
- `Week2` notebooks: introductory material, not focused on agent/RAG/tool workflows

## 1. Week 7: Prompt Engineering Foundations

Source: `Labs/Week7/05_prompt_engineering_tutorial_new.ipynb`

### Core principles introduced

1. Write clear and specific instructions.
2. Give the model time to think.

### Tactics introduced in class

- Use delimiters to clearly separate different parts of the input.
  - Examples in class: triple backticks, triple quotes, tags, explicit separators.
  - Assignment relevance: separate witness transcript, town record, tool result, and task instructions so the model does not mix them up.
- Ask for structured output.
  - Examples in class: JSON or HTML output.
  - Assignment relevance: ask the model to return suspect lists, hypotheses, extracted clues, or evidence tables in JSON for downstream processing.
- Ask the model to check whether conditions are satisfied.
  - Class pattern: detect whether a text contains instructions, otherwise return `"No steps provided."`
  - Assignment relevance: ask the model to verify whether an alibi is supported by evidence, whether a clue is relevant, or whether there is enough evidence to accuse someone.
- Few-shot prompting.
  - Class pattern: give one style/example pair, then ask the model to continue in the same style.
  - Assignment relevance: provide one example of a good evidence analysis or clue extraction format, then have the model process new transcripts consistently.
- Specify the steps required to complete a task.
  - Class pattern: multi-step instruction list such as summarize, translate, extract names, output JSON.
  - Assignment relevance: explicitly tell the model to identify entities, compare records, check contradictions, then propose next queries.
- Instruct the model to work out its own solution before giving a conclusion.
  - Class pattern: reasoning-first prompting to avoid premature answers.
  - Assignment relevance: have the model reason through witness statements and record checks before naming a suspect.

### Practical capabilities covered

- summarizing
- inferring
- transforming
- expanding
- chatbot-style interaction

### What to reuse for Assignment 5

- Strong system prompts
- explicit output schemas
- step-by-step decomposition
- evidence separation with delimiters
- prompts that force verification before conclusion

## 2. Week 8: CoT, Prompt Chaining, Simple Retrieval, and Output Analysis

Primary source: `Labs/Week8/07_llm functional prototype.ipynb`

### 2.1 Chain-of-Thought / step-by-step reasoning

The notebook builds a service agent and explicitly teaches:

- chain of thoughts
- chaining prompts
- simple retrieval / RAG-style flow
- consistent solving of complex tasks

### CoT pattern shown in class

The notebook uses a long system prompt with a required multi-step format:

- Step 1: decide whether the user asks about a specific product
- Step 2: check whether the product exists in allowed data
- Step 3: list user assumptions
- Step 4: verify whether assumptions are true
- Step 5: respond to the user

Assignment relevance:

- apply the same pattern to suspects, witnesses, locations, and records
- force the model to separate observation, assumption, verification, and final conclusion
- reduce hallucination by requiring each conclusion to come from provided evidence

### Inner monologue / hiding reasoning

The notebook uses a delimiter such as `####` to separate reasoning steps and then extracts only the final user-facing answer.

Assignment relevance:

- keep internal reasoning or analysis traces separate from the final report
- show clean final outputs while preserving intermediate reasoning in logs if needed

### 2.2 Chaining prompts

The notebook demonstrates a multi-stage pipeline:

1. First LLM call extracts relevant products/categories from user text.
2. Python parses the structured output.
3. Code retrieves detailed records from a local dictionary.
4. A second LLM call answers using only the retrieved information.

This is extremely relevant to the detective assignment.

Equivalent detective version:

1. extract suspect names / locations / dates / vehicles from an interview
2. parse to Python objects
3. retrieve matching rows from town records
4. ask the model to answer with the retrieved evidence only

### 2.3 Simple retrieval / early RAG-style design

The notebook explicitly notes that the retrieved product info is joined with structured data in a "simple retrieval / RAG-style flow."

Key pattern:

- LLM extracts what to look up
- code retrieves the ground-truth data
- LLM answers based on that retrieved evidence

This is a simple but strong architecture for Assignment 5 even before full vector RAG.

### 2.4 Analyzing model outputs

The notebook includes an evaluation prompt where another LLM checks:

- whether the answer sufficiently addresses the user question
- whether the facts used are correct with respect to the retrieved data

This directly supports the rubric item "Analyze the output of LLMs."

Assignment relevance:

- add an evaluator prompt to check whether the detective agent cited real evidence
- reject outputs that mention unsupported gym visits, car plates, or witness statements
- use a Y/N or structured rubric for groundedness and sufficiency

## 3. Week 8: Tool Use / Function Calling

Primary source: `Labs/Week8/07_llm_funtion_call.ipynb`

### What was introduced

The notebook explicitly teaches the full function-calling workflow:

1. define a tool schema
2. send tool schema with the user message
3. let the model request a tool call
4. execute the Python function yourself
5. send the tool result back to the model
6. get the final answer

### Key concepts from class

- The model does not run Python by itself.
- The model returns structured intent through `tool_calls`.
- Your code must execute the real function.
- Your code must append the tool result as a `tool` message.
- Then the model can produce the final natural-language answer.

### Tool schema design

The notebook teaches a JSON-style schema with:

- function name
- description
- typed parameters
- `required`
- `additionalProperties: false`

Assignment relevance:

You can define tools such as:

- `lookup_witness_by_address`
- `search_interview_by_name`
- `lookup_gym_checkins`
- `lookup_vehicle_log`
- `lookup_social_event_attendance`
- `lookup_license_plate`

### Tool routing and multi-tool workflows

The notebook also shows:

- a dispatch table from tool name to Python function
- an execution loop that keeps running until no more tool calls remain
- multi-tool workflows where one tool result triggers another tool call

Assignment relevance:

- the agent can first extract a partial plate, then query vehicle logs, then query the suspect's interview
- the agent can read a witness clue, query records, then query another tool based on the result

### Controls introduced in class

- `tool_choice="none"` to disable tool use
- force a specific tool with `tool_choice={...}`
- maintain logs of tool calls for debugging

### Good practices taught

- clear tool names
- strong descriptions
- simple parameter schemas
- validate arguments before running tools
- do not let the model directly perform unsafe operations
- log tool calls for debugging

These are directly reusable in the detective system.

## 4. Week 9: Full RAG with MongoDB

Primary source: `Labs/Week9/09_RAGwithMongoDB.ipynb`

### Core RAG components introduced in class

The notebook demonstrates a standard RAG pipeline:

1. store documents and embeddings in a database
2. embed the user's query
3. run vector similarity search
4. retrieve the most relevant records
5. provide the retrieved context to the LLM
6. ask the LLM to answer using that context

### Embeddings

The notebook uses:

- `text-embedding-3-small`
- vector size `1536`

The class material shows how text is transformed into embeddings and stored in a field like `text_embeddings`.

### Vector search index

The notebook defines a MongoDB Atlas vector index:

- field: `text_embeddings`
- dimensions: `1536`
- similarity: `cosine`
- type: `knnVector`

This is the retrieval infrastructure behind semantic search.

### Query embedding and vector search

The notebook defines:

- `get_embedding(text)` to embed a query
- `vector_search(user_query, db, collection, vector_index="vector_index_text")`

The vector search stage includes:

- the index name
- the query vector
- the embedding field path
- candidate count
- result limit

### End-to-end query handling

The notebook's `handle_user_query(...)` shows the full RAG flow:

1. take plain-text user query
2. retrieve relevant documents via vector search
3. format results into a table / structured context
4. send question plus retrieved context to the LLM
5. generate an answer grounded in retrieved data

### What to reuse for Assignment 5

RAG is especially suitable for:

- long interview transcripts
- police notes
- witness statements
- town reports
- any large text corpus where exact keyword search is not enough

Recommended detective use:

- chunk each transcript / record file
- embed each chunk
- retrieve top relevant chunks for a question like "Who mentioned a red car?" or "What evidence links Annabel to the suspect?"
- answer only from those retrieved chunks

### Important class takeaway

RAG is not just "ask the LLM with a long prompt."
It is:

- retrieval first
- grounded context second
- generation last

This is one of the main required rubric items for the assignment.

## 5. Week 10: Agentic Workflow, Reflection, Planning, and Multi-Agent

Primary source: `Labs/Week10/Lab09_MultiAgent_and_Human_AI_Cooperation.ipynb`

This part is most relevant for the optional extra-credit multi-agent design/implementation.

### 5.1 Agentic workflow design

The notebook introduces a general agent loop:

`Perception -> Reasoning -> Action -> Observation/Feedback`

Definitions from class:

- Perception: receive user input, file, or webpage
- Reasoning: break the goal into subtasks
- Action: use tools or generate output
- Observation/Feedback: evaluate results and loop if needed

This is a strong conceptual template for the detective agent.

### 5.2 Reflection

The notebook covers:

- no reflection
- self-reflection
- external reflection

Key class takeaway:

- for objective tasks such as code, math, or data processing, external reflection is more reliable
- for more subjective tasks, self-reflection can still help

Assignment relevance:

- external reflection can mean checking suspect claims against real database/tool outputs
- a critic agent or evaluator prompt can verify that every accusation is grounded

### 5.3 Planning agent

The notebook introduces building an agent that constructs and executes its own plan.

Assignment relevance:

- the system can generate an investigation plan such as:
  - identify witnesses
  - retrieve their interviews
  - extract names/plates/locations
  - query town records
  - compare alibis
  - identify killer
  - identify mastermind

### 5.4 Multi-agent systems

The notebook introduces:

- manager pattern
- group chat pattern
- AutoGen-based multi-agent coordination

Possible detective decomposition based on class concepts:

- `Witness Analyst Agent`
- `Records Lookup Agent`
- `Hypothesis Critic Agent`
- `Case Manager Agent`

This is optional, but useful if you want extra points.

## 6. Week 11: Skills, Subagents, and MCP

Primary sources:

- `Labs/Week11/Lab10_Skills_and_MCP.ipynb`
- `Labs/Week11/design_mcp_server.py`

### 6.1 Skills

Week 11 adds an important layer on top of prompting: instead of rewriting long prompts every time, package them as reusable **Skills**.

Key class ideas:

- ad-hoc prompts do not scale
- skills are reusable structured instruction sets
- a `SKILL.md` contains frontmatter plus markdown instructions
- skills load on demand, so they reduce context bloat

Assignment relevance:

- create a skill for `witness-analysis`
- create a skill for `alibi-check`
- create a skill for `case-summary`
- create a skill for `final-accusation-report`

This is useful if your detective workflow repeats the same analysis pattern across many interview files.

### 6.2 Skills + subagents

Week 11 extends Week 10 by showing that skills can be composed with **subagents** for isolated execution.

Key class takeaway:

- subagents keep context clean by isolating heavy exploration or analysis
- only the summary/result returns to the main conversation
- each skill can run in its own isolated context

Assignment relevance:

- run one subagent to analyze witness interviews
- run another to analyze structured records
- run another to challenge unsupported hypotheses
- keep the main case manager focused on the final synthesis

This is especially helpful if the case data becomes large and noisy.

### 6.3 MCP as a tool standard

Week 11 also introduces **MCP (Model Context Protocol)** as a universal way to expose tools.

Core class points:

- MCP standardizes tools across model providers
- an MCP server exposes tools, resources, and prompts
- clients can discover tools automatically
- tool schemas can be bridged into function-calling format

The lab explicitly shows a client workflow that:

1. connects to an MCP server
2. discovers available tools automatically
3. converts tool schemas into model-callable format
4. runs an LLM/tool loop
5. routes tool calls through MCP

Assignment relevance:

- your detective tools do not have to be plain local Python functions only
- you could expose record lookup, transcript search, or evidence retrieval through an MCP server
- this makes the system more modular and closer to a real agent-tool architecture

Potential detective MCP tools:

- `search_case_transcripts`
- `lookup_license_record`
- `lookup_vehicle_logs`
- `lookup_gym_attendance`
- `lookup_event_attendance`
- `get_case_graph_neighbors`

### 6.4 Skills vs MCP

Week 11 makes an important distinction:

- **Skills** = reusable instructions / workflows
- **MCP tools** = actual capabilities / computations / data access

For Assignment 5, this maps cleanly to:

- use **skills** for how the detective should analyze evidence
- use **MCP tools** or function-calling tools for accessing the actual records

### 6.5 What Week 11 adds beyond earlier weeks

Compared with Weeks 7 to 10, Week 11 contributes:

- reusable packaged workflows instead of one-off prompts
- cleaner orchestration using skill pipelines
- stronger modularity through subagent isolation
- a more general and portable tool architecture through MCP

This is not strictly required by the assignment rubric, but it can make your implementation more advanced and better structured.

## 7. Best-Matching Course Concepts for the Detective Assignment

If you want the assignment to align closely with the class material, the strongest course-to-assignment mapping is:

### A. Prompting

Use Week 7 techniques:

- clear task instructions
- delimiters
- structured JSON outputs
- step-by-step prompts
- prompts that force checking before concluding

### B. CoT

Use Week 8's step-by-step reasoning pattern:

- identify clue
- identify assumptions
- verify with records
- state what is supported vs unsupported
- propose next action

### C. Tool use

Use Week 8 function-calling patterns:

- expose lookup tools to the model
- let the model choose the correct tool
- execute tools in Python
- feed results back into the conversation

### D. RAG

Use Week 9's retrieval pattern:

- embed transcript chunks
- retrieve relevant interview sections
- answer from retrieved context only

### E. Output analysis

Use Week 8 evaluation patterns and Week 10 reflection:

- check whether final answers are grounded
- check whether each claim is supported by retrieved/tool evidence
- ask the system to refuse unsupported accusations

### F. Optional multi-agent

Use Week 10 if you want extra credit:

- one manager agent
- several specialized worker agents
- a critic/evaluator agent

## 8. Recommended Design Direction Based on Class Material

A course-aligned detective system could look like this:

1. Use RAG over interview transcripts.
2. Use tool calling for structured record lookup.
3. Use CoT-style prompts for hypothesis building.
4. Use an evaluator step to reject unsupported conclusions.
5. Optionally package repeated workflows as skills.
6. Optionally add a manager + worker multi-agent architecture.
7. Optionally expose lookup capabilities through MCP.

## 9. Most Reusable Patterns from Class

These are the highest-value patterns to directly copy into Assignment 5:

- delimiter-based prompts
- structured JSON outputs
- step-by-step reasoning prompts
- extraction -> parsing -> retrieval -> final answer pipeline
- tool schema + tool router + tool loop
- RAG pipeline with embeddings and vector search
- evaluator / critic pass for groundedness
- agent loop: perceive, reason, act, observe
- reusable skills for repeated workflows
- MCP tool abstraction for modular data access

## 10. Bottom Line

If you want your assignment to clearly reflect class concepts, the most important notebooks are:

1. `Week7/05_prompt_engineering_tutorial_new.ipynb`
2. `Week8/07_llm functional prototype.ipynb`
3. `Week8/07_llm_funtion_call.ipynb`
4. `Week9/09_RAGwithMongoDB.ipynb`
5. `Week10/Lab09_MultiAgent_and_Human_AI_Cooperation.ipynb` for optional extra credit
6. `Week11/Lab10_Skills_and_MCP.ipynb` for reusable skills, subagent isolation, and MCP-based tool architecture

The most important concepts to demonstrate are:

- careful prompting
- explicit step-by-step reasoning
- grounded retrieval
- tool calling
- evidence checking / output evaluation
- optional multi-agent orchestration
- optional reusable skills
- optional MCP-based tool integration