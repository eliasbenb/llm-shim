# llm-shim

OpenAI-compatible API shim for routing chat and embeddings requests to one or more LLM providers via [Instructor](https://python.useinstructor.com/).

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/embeddings`

## Requirements

- Python `>=3.14`
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- API keys for any providers you configure

## Quick start

1. Install dependencies:

```bash
uv sync
```

2. Create local environment config:

```bash
cp .env.example .env
```

3. Edit `.env` with your model + API key.

4. Start server:

```bash
uv run main.py
```

5. Open docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

Settings use `LLM_SHIM_` prefix and nested keys with `__`.

Required minimum:

```dotenv
LLM_SHIM_DEFAULT_PROVIDER=default

LLM_SHIM_PROVIDERS__default__MODEL=openai/gpt-4.1-mini
LLM_SHIM_PROVIDERS__default__API_KEY=sk-...

LLM_SHIM_SERVER__HOST=0.0.0.0
LLM_SHIM_SERVER__PORT=8000
LLM_SHIM_SERVER__RELOAD=false
LLM_SHIM_SERVER__WORKERS=1
LLM_SHIM_SERVER__LOG_LEVEL=info
```

Model values must use `provider/model` format, for example `openai/gpt-4.1-mini`.

## Model routing rules

Incoming `model` is resolved in this order:

1. If omitted: use `LLM_SHIM_DEFAULT_PROVIDER`
2. If it matches a configured provider alias: use that provider
3. If it exactly matches a configured provider model string: use that provider
4. Otherwise: return `400`

## API examples

### Chat completion

```bash
curl -s http://localhost:8000/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d '{
		"model": "default",
		"messages": [
			{"role": "user", "content": "Testing..."}
		]
	}'
```

### Chat completion with JSON schema output

```bash
curl -s http://localhost:8000/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d '{
		"model": "default",
		"messages": [
			{"role": "user", "content": "Return a short summary and confidence."}
		],
		"response_format": {
			"type": "json_schema",
			"json_schema": {
				"name": "summary_response",
				"schema": {
					"type": "object",
					"properties": {
						"summary": {"type": "string"},
						"confidence": {"type": "number"}
					},
					"required": ["summary", "confidence"]
				}
			}
		}
	}'
```

`content` is returned as a JSON string for schema-based responses.

### Embeddings

```bash
curl -s http://localhost:8000/v1/embeddings \
	-H 'Content-Type: application/json' \
	-d '{
		"model": "default",
		"input": "llm-shim"
	}'
```
