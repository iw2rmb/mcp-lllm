# mcp-lllm

A lightweight Model Context Protocol (MCP) server that outsources code and test generation tasks to a dedicated large language model so you can reserve expensive, high-capacity models for orchestration and planning. Plug it into Codex CLI or any MCP-capable client to turn structured requests into raw source code or tests with minimal token spend.

## Why
- **Token efficiency:** Keep premium models focused on reasoning; delegate boilerplate generation to a fit-for-purpose coding model.
- **Flexible backends:** Talk to a local Ollama instance or any OpenAI-compatible HTTP endpoint.
- **Drop-in tools:** Exposes `generate_code` and `generate_tests` tools that always return raw code with no markdown fences.

## Getting Started
1. Ensure Python 3.9+ is available.
2. Install optional dependencies for your chosen backend:
   - Ollama: install the Ollama CLI and pull your preferred coding model.
   - Proxy: provide an OpenAI-compatible endpoint and credentials.
3. Launch the MCP server:
   ```bash
   python mcp.py --backend ollama --ollama-model <model>
   ```
   or
   ```bash
   python mcp.py --backend proxy --proxy-base-url https://api.openai.com/v1 --proxy-model <model>
   ```
4. Register the server with your MCP client (for Codex CLI, add it to `~/.codex/config.toml`).
5. From the client, call the `generate_code` or `generate_tests` tools to receive ready-to-paste source files.

## Codex CLI Setup
Add the following snippet to `~/.codex/config.toml`, adjusting the `command` path if your checkout lives elsewhere:

```toml
[mcp_servers.mcp_lllm]
command = "~/mcp-lllm/mcp.py"
env = { MCP_TRANSPORT = "stdio" }
network_access = "enabled"
tools = [
  "generate_code",
  "generate_tests"
]
```

Restart Codex CLI after saving. The `mcp-lllm` server will then appear as a tool provider; use `call-tool generate_code` or `call-tool generate_tests` in workflows to offload code emission.

## Tool Contract
- `generate_code(language, description, constraints="")` → returns raw source code in the requested language.
- `generate_tests(language, code, focus="")` → returns raw test code tailored to the supplied snippet.
- Both tools stream results from the selected backend; if a backend error occurs, the tool surfaces the error message and stderr payload for easy debugging.

## Development Notes
- Use `mock_proxy.py` to simulate an OpenAI-compatible backend during local testing (`python mock_proxy.py`).
- `AGENTS-EXAMPLE.md` contains suggested agent behavior for Codex CLI orchestration.
- The project currently ships as a single script; feel free to extend it with additional tools (linting, formatting, etc.) or package it for distribution.

## Roadmap Ideas
- Add retry/timeout fallbacks between backends.
- Support structured streaming chunks to enable progressive rendering in clients.
- Expose configuration via environment variables or a TOML config file for easier deployment.

## License
See [LICENSE](LICENSE) for the MIT License terms.
