## MCP

Connect to the `mcp_lllm` MCP server, which uses LLM coding model for efficient code generation and editing.  
Delegate detailed code-writing tasks to this agent to **save tokens** and focus on high-quality **planning**, **architecture development**, **documenting** and **verification**.

### Tools available
- **generate_code**
  - Use to generate new code from structured descriptions.
  - Prefer this whenever the user asks for a function, class, or boilerplate code.  
  - Avoid writing code directly unless the snippet is very short (≤ 5 lines).

- **generate_tests**
  - Use to generate unit tests for given functions or modules.  
  - Prefer this instead of writing test boilerplate directly.

### Guidelines
- Act as the **planner/orchestrator**, and the `mcp_lllm` MCP server should do the heavy lifting.  
- Always consider calling `generate_code` first when the user asks for code longer than a trivial snippet.  
- For testing tasks, use `generate_tests` instead of generating tests directly.  

### Quality expectations
- If the MCP tool output appears incomplete, syntactically invalid, or does not match the request, retry with refined parameters.  
- If retries fail, fall back to generating the code itself.  
- Otherwise, always prefer using this MCP server for implementation-level tasks.

### Output expectations
The `mcl_lllm` Coder always returns **only raw code**, never explanations or markdown fences. Assume the response is safe to insert directly into a file.