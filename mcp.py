#!/usr/bin/env python3
import argparse
import json
import os
import selectors
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_OLLAMA_MODEL = "freehuntx/qwen3-coder:8b"
CONFIG = None


# ---------- CLI Configuration ----------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MCP agent for local or OpenAI-compatible LLM backends"
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "proxy"],
        default="ollama",
        help="Select LLM backend: 'ollama' (local) or 'proxy' (OpenAI-compatible HTTP endpoint)",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Model identifier to pass to Ollama",
    )
    parser.add_argument(
        "--proxy-base-url",
        help="Base URL for the OpenAI-compatible endpoint (e.g. http://localhost:8000 or https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--proxy-model",
        help="Model identifier to request from the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--proxy-api-key",
        help="API key for the proxy; defaults to OPENAI_API_KEY env var",
    )
    parser.add_argument(
        "--proxy-organization",
        help="Optional organization header for the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--proxy-timeout",
        type=float,
        default=60.0,
        help="Request timeout (seconds) for the OpenAI-compatible endpoint",
    )
    return parser.parse_args(argv)


# ---------- Utility ----------
def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def recv():
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)


def run_ollama(prompt, model=DEFAULT_OLLAMA_MODEL):
    """Call Ollama locally for code generation using streaming reads."""
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return json.dumps({
            "error": "Ollama binary not found. Ensure it is installed and in PATH."
        })
    except Exception as exc:
        return json.dumps({"error": f"Error launching Ollama: {exc}"})

    stdout_chunks = []
    stderr_chunks = []

    try:
        if prompt:
            process.stdin.write(prompt.encode("utf-8"))
        process.stdin.close()

        selector = selectors.DefaultSelector()
        if process.stdout:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        if process.stderr:
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")

        while selector.get_map():
            for key, _ in selector.select(timeout=0.1):
                data = key.fileobj.read(4096)
                if not data:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                if key.data == "stdout":
                    stdout_chunks.append(data)
                else:
                    stderr_chunks.append(data)

        return_code = process.wait()
    except Exception as exc:
        process.kill()
        process.wait()
        return json.dumps({"error": f"Error during Ollama call: {exc}"})

    stdout_text = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

    if return_code != 0:
        return json.dumps({
            "error": "Ollama call failed",
            "stderr": stderr_text
        })

    return stdout_text


def normalize_proxy_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions") or base.endswith("/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if base.endswith("/v1/"):
        return f"{base}chat/completions"
    if base.endswith("/v1/chat"):
        return f"{base}/completions"
    if base.endswith("/v1/chat/"):
        return f"{base}completions"
    return f"{base}/v1/chat/completions"


def run_openai_proxy(
    prompt,
    *,
    base_url,
    model,
    api_key=None,
    organization=None,
    timeout=60.0,
):
    """Call an OpenAI-compatible endpoint using the Chat Completions API."""
    if not base_url:
        return json.dumps({"error": "Proxy base URL not configured."})
    if not model:
        return json.dumps({"error": "Proxy model not configured."})

    url = normalize_proxy_url(base_url)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if organization:
        headers["OpenAI-Organization"] = organization

    request_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(request_payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", response.getcode())
            body_bytes = response.read()
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        return json.dumps({
            "error": f"Proxy call failed with status {exc.code}",
            "stderr": err_body,
        })
    except Exception as exc:
        return json.dumps({"error": f"Proxy call failed: {exc}"})

    body_text = body_bytes.decode("utf-8", errors="replace")

    try:
        response_payload = json.loads(body_text)
    except json.JSONDecodeError:
        return json.dumps({"error": "Proxy returned invalid JSON", "stderr": body_text})

    if status >= 400:
        return json.dumps({
            "error": f"Proxy call failed with status {status}",
            "stderr": body_text,
        })

    choices = response_payload.get("choices") if isinstance(response_payload, dict) else None
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if content is not None:
                    return content
            text = first_choice.get("text")
            if text is not None:
                return text

    return json.dumps({"error": "Proxy response missing content", "stderr": body_text})


def call_model(prompt):
    """Dispatch model calls according to the configured backend."""
    if CONFIG is None:
        model = DEFAULT_OLLAMA_MODEL
        return run_ollama(prompt, model=model), {"backend": "ollama", "model": model}

    backend = CONFIG.backend
    if backend == "ollama":
        model = CONFIG.ollama_model or DEFAULT_OLLAMA_MODEL
        return run_ollama(prompt, model=model), {"backend": "ollama", "model": model}

    if backend == "proxy":
        result = run_openai_proxy(
            prompt,
            base_url=CONFIG.proxy_base_url,
            model=CONFIG.proxy_model,
            api_key=CONFIG.proxy_api_key,
            organization=CONFIG.proxy_organization,
            timeout=CONFIG.proxy_timeout,
        )
        return result, {
            "backend": "proxy",
            "model": CONFIG.proxy_model,
            "base_url": CONFIG.proxy_base_url,
        }

    return json.dumps({"error": f"Unsupported backend: {backend}"}), {"backend": backend}


def clean_code_output(text):
    # Strip markdown code fences like ```rust ... ```
    if text.strip().startswith("```"):
        parts = text.split("```")
        # Take the middle part (inside the fences)
        if len(parts) >= 2:
            inner = parts[1]
            # If it starts with a language tag (e.g. "rust\n"), drop it
            if "\n" in inner:
                inner = inner.split("\n", 1)[1]
            return inner.strip()
    return text.strip()


def prepare_patch_input(patch_text: str, filepath: Path):
    raise NotImplementedError("apply_patch tool has been removed")


def make_text_response(text: str, is_error: bool = False, metadata: dict = None):
    response = {"content": [{"type": "text", "text": text}], "code": text, "isError": bool(is_error)}
    if metadata is not None:
        response["metadata"] = metadata
    return response


# ---------- Tool Handlers ----------
def handle_generate_code(params):
    """Generate code from a structured task description"""
    lang = params.get("language", "python")
    desc = params.get("description", "")
    constraints = params.get("constraints", "")

    prompt = (
        f"You are a coding assistant.\n"
        f"Write {lang} code for the following task.\n"
        f"### Task\n{desc}\n\n"
        f"### Constraints\n{constraints}\n\n"
        f"### Output rules\n"
        f"- Return ONLY valid {lang} source code.\n"
        f"- Do not include explanations, markdown fences, or extra text.\n"
        f"- Output must be directly usable as {lang} code.\n"
    )

    raw, metadata = call_model(prompt)
    code = clean_code_output(raw)

    try:
        parsed = json.loads(code)
    except json.JSONDecodeError:
        parsed = None
    except TypeError:
        parsed = None

    if isinstance(parsed, dict) and "error" in parsed:
        error_message = parsed.get("error") or json.dumps(parsed)
        if parsed.get("stderr"):
            error_message += f"\n{parsed['stderr']}"
        return make_text_response(error_message, is_error=True)

    if isinstance(code, str) and code.lower().startswith("error:"):
        return make_text_response(code, is_error=True)

    return make_text_response(code.strip(), metadata=metadata)


EXTS = {
    "python": "*.py",
    "rust": "*.rs",
    "javascript": "*.js",
    "typescript": "*.ts",
    "cpp": "*.cpp",
    "c": "*.c",
    "go": "*.go"
    # add more as needed
}


def handle_generate_tests(params):
    """Generate unit tests for given code snippet"""
    lang = params.get("language", "python")
    code = params.get("code", "")
    focus = params.get("focus", "")

    prompt = (
        f"Write {lang} unit tests for the following code.\n"
        f"Focus: {focus}\n"
        f"--- CODE START ---\n{code}\n--- CODE END ---"
    )
    raw, metadata = call_model(prompt)
    tests = clean_code_output(raw)

    try:
        parsed = json.loads(tests)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, dict) and "error" in parsed:
        error_message = parsed.get("error") or json.dumps(parsed)
        if parsed.get("stderr"):
            error_message += f"\n{parsed['stderr']}"
        return make_text_response(error_message, is_error=True)

    if isinstance(tests, str) and tests.lower().startswith("error:"):
        return make_text_response(tests, is_error=True)

    return make_text_response(tests.strip(), metadata=metadata)


# ---------- Dispatch ----------
TOOLS = {
    "generate_code": handle_generate_code,
    "generate_tests": handle_generate_tests,
}

# --- Define tool schemas once ---
TOOL_SCHEMAS = {
    "generate_code": {
        "description": "Generate code from a structured task description",
        "inputSchema": {
            "type": "object",
            "properties": {
                "language": {"type": "string"},
                "description": {"type": "string"},
                "constraints": {"type": "string"}
            },
            "required": ["language", "description"]
        }
    },
    "apply_patch": {
        "description": "Apply a unified diff patch to a file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "patch": {"type": "string"}
            },
            "required": ["filepath", "patch"]
        }
    },
    "generate_tests": {
        "description": "Generate unit tests for given code snippet",
        "inputSchema": {
            "type": "object",
            "properties": {
                "language": {"type": "string"},
                "code": {"type": "string"},
                "focus": {"type": "string"}
            },
            "required": ["language", "code"]
        }
    }
}


def handle_request(req):
    method = req.get("method")

    # --- Initialization handshake ---
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "mcp-lllm",
                "version": "0.1.0"
            },
            "capabilities": {
                "tools": {"list": {}, "call": {}},
                "resources": {"list": {}},
                "prompts": {"list": {}},
                "logging": {"list": {}}
            }
        }

    # --- Listing tools (Codex will call this) ---
    if method == "tools/list":
        return {
            "tools": [
                {"name": name, **spec}
                for name, spec in TOOL_SCHEMAS.items()
            ]
        }

    # --- Handling tool calls ---
    if method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if tool_name in TOOLS:
            try:
                return TOOLS[tool_name](arguments)
            except Exception as e:
                return {"error": str(e)}
        return {"error": f"Unknown tool: {tool_name}"}

    if method and method.startswith("tools/"):
        tool = method.split("/", 1)[1]
        if tool in TOOLS:
            try:
                return TOOLS[tool](req.get("params", {}))
            except Exception as e:
                return {"error": str(e)}

    return {"error": f"Unknown method: {method}"}


# ---------- Main Loop ----------
def main():
    global CONFIG
    CONFIG = parse_args()

    if not getattr(CONFIG, "proxy_api_key", None):
        CONFIG.proxy_api_key = os.environ.get("OPENAI_API_KEY")
    if not getattr(CONFIG, "proxy_organization", None):
        CONFIG.proxy_organization = os.environ.get("OPENAI_ORGANIZATION")
    if not getattr(CONFIG, "ollama_model", None):
        CONFIG.ollama_model = DEFAULT_OLLAMA_MODEL

    while True:
        req = recv()
        if req is None:
            break
        result = handle_request(req)
        response = {
            "id": req.get("id", str(uuid.uuid4())),
            "jsonrpc": "2.0",
        }

        if isinstance(result, dict) and "error" in result and "content" not in result and "tools" not in result:
            error_obj = {"code": -32000, "message": str(result.get("error"))}
            details = result.get("details")
            if details is not None:
                error_obj["data"] = details
            response["error"] = error_obj
        else:
            response["result"] = result

        send(response)


if __name__ == "__main__":
    main()
