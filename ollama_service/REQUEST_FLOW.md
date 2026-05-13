# Ollama Service Request Flow with Logging

## Visual Request Processing Flow

### 1. Chat Request Processing Flow

```
Client Request (POST /v1/chat)
        │
        ├─→ [1] API Key Verification (verify_api_key)
        │   ├─ NO KEY → HTTP 401, WARNING log
        │   ├─ INVALID → HTTP 403, WARNING log
        │   └─ OK → DEBUG log
        │
        ├─→ [2] INFO Log: "request received messages=N"
        │
        ├─→ [3] Rate Limit Check (enforce_rate_limit)
        │   ├─ DEBUG log: "step=enforce_rate_limit"
        │   ├─ DEBUG log in security.py: "count=N/LIMIT"
        │   ├─ LIMIT EXCEEDED → HTTP 429, WARNING log
        │   └─ OK → DEBUG log: "step=rate_limit_ok elapsed_ms=X.XX"
        │
        ├─→ [4] Input Validation (_validate_messages)
        │   ├─ DEBUG log: "step=validate_messages"
        │   ├─ TOO MANY MESSAGES → HTTP 413, error logged
        │   ├─ TEXT TOO LONG → HTTP 413, error logged
        │   └─ OK → DEBUG log: "step=validate_ok messages=N elapsed_ms=X.XX"
        │
        ├─→ [5] Payload Construction
        │   ├─ DEBUG log: "step=construct_payload model=MODEL"
        │   └─ DEBUG log: "step=payload_ready num_messages=N elapsed_ms=X.XX"
        │
        ├─→ [6] Upstream Call (post_ollama)
        │   ├─ DEBUG log: "step=call_upstream timeout_seconds=120"
        │   ├─ INFO log (ollama_client.py): "-> upstream POST /api/chat"
        │   │
        │   ├─ [NETWORK CALL TO OLLAMA]
        │   │
        │   ├─ TIMEOUT → ERROR log, HTTP 504
        │   ├─ CONNECTION ERROR → ERROR log, HTTP 502
        │   ├─ HTTP ERROR → ERROR log, HTTP 400+
        │   └─ SUCCESS → INFO log (ollama_client.py): "upstream POST ok elapsed_ms=X.XX"
        │
        ├─→ [7] Response Preparation
        │   └─ INFO log: "response sent model=MODEL elapsed_ms=X.XX"
        │
        └─→ Client Response (HTTP 200 + data)
```

### 2. Embedding Request Processing Flow

```
Client Request (POST /v1/embed)
        │
        ├─→ [API Key + Rate Limit] (same as chat)
        │
        ├─→ INFO Log: "request received batch_size=N"
        │
        ├─→ Input Validation
        │   ├─ DEBUG log: "step=validate_batch batch_size=N max_chars=LIMIT"
        │   ├─ TOO MANY ITEMS → HTTP 413, error logged
        │   ├─ TEXT TOO LONG → HTTP 413, error logged
        │   └─ OK → DEBUG log: "step=validate_ok elapsed_ms=X.XX"
        │
        ├─→ Payload Construction
        │   ├─ DEBUG log: "step=construct_payload model=EMBEDDING_MODEL"
        │   └─ DEBUG log: "step=payload_ready elapsed_ms=X.XX"
        │
        ├─→ Timing Start
        │   └─ INFO log: "[timing] step=generate_embedding status=start batch_size=N input_chars=TOTAL"
        │
        ├─→ Upstream Call
        │   ├─ DEBUG log: "step=call_upstream timeout_seconds=60"
        │   ├─ INFO log (ollama_client.py): "-> upstream POST /api/embed"
        │   │
        │   ├─ [NETWORK CALL TO OLLAMA]
        │   │
        │   └─ SUCCESS → INFO log (ollama_client.py): "upstream POST ok elapsed_ms=X.XX"
        │
        ├─→ Timing End
        │   └─ INFO log: "[timing] step=generate_embedding status=ok output_vectors=COUNT elapsed_ms=X.XX"
        │
        └─→ Client Response (HTTP 200 + embeddings)
```

## Logging Timeline Example

### Successful Chat Request (1543ms total)

```
Time    Source          Message
────────────────────────────────────────────────────────────
0ms     app.main        [chat] request received messages=2
0ms     app.main        step=enforce_rate_limit
0ms     app.security    [ratelimit] count=1/60
0.45ms  app.main        step=rate_limit_ok elapsed_ms=0.45
0.45ms  app.main        step=validate_messages
0.52ms  app.main        step=validate_ok messages=2 elapsed_ms=0.52
0.52ms  app.main        step=construct_payload model=llama2
0.89ms  app.main        step=payload_ready num_messages=2 elapsed_ms=0.89
0.89ms  app.main        step=call_upstream timeout_seconds=120
0.89ms  app.client      -> upstream POST /api/chat
        [... NETWORK LATENCY: ~1543ms ...]
1543ms  app.client      upstream POST ok status=200 elapsed_ms=1542.89
1544ms  app.main        response sent model=llama2 elapsed_ms=1543.56
```

### Analysis
- **Local processing**: 0.89ms (0.06%)
- **Upstream call**: 1542.89ms (99.94%)
- **Total**: 1543.56ms

## Error Handling Flow

### Invalid API Key
```
Client Request
    ├─→ verify_api_key()
    ├─→ WARNING log: "invalid api key attempt"
    └─→ HTTP 403 Forbidden (FAST: <1ms)
```

### Rate Limit Exceeded
```
Client Request
    ├─→ INFO log: "request received"
    ├─→ enforce_rate_limit()
    ├─→ DEBUG log: "count=61/60"
    ├─→ WARNING log: "limit exceeded"
    └─→ HTTP 429 Too Many Requests (FAST: ~1ms)
```

### Input Validation Error
```
Client Request
    ├─→ Security checks OK
    ├─→ Validation function called
    ├─→ Constraint violated (e.g., text too long)
    ├─→ ERROR logged by validation function
    └─→ HTTP 413 Payload Too Large (FAST: ~1ms)
```

### Upstream Timeout
```
Client Request
    ├─→ All local checks pass
    ├─→ INFO log: "step=call_upstream"
    ├─→ Sends to Ollama
    ├─→ 120s timeout elapsed
    ├─→ ERROR log: "upstream timeout elapsed_ms=120000"
    └─→ HTTP 504 Gateway Timeout (SLOW: 120+ seconds)
```

### Connection Error
```
Client Request
    ├─→ All local checks pass
    ├─→ INFO log: "step=call_upstream"
    ├─→ Attempts to connect to Ollama
    ├─→ Connection refused
    ├─→ ERROR log: "upstream request error [errno]"
    └─→ HTTP 502 Bad Gateway (FAST: ~50ms)
```

## Log Component Hierarchy

```
Logger: root (Level: INFO)
├── app.main (FILE HANDLER)
│   ├── [ollama-service][health] - Health checks
│   ├── [ollama-service][ready] - Readiness checks
│   ├── [ollama-service][models] - Model listing
│   ├── [ollama-service][chat] - Chat endpoint
│   ├── [ollama-service][api_chat] - Native chat
│   ├── [ollama-service][generate] - Generation endpoint
│   ├── [ollama-service][api_generate] - Native generation
│   ├── [ollama-service][embed] - Embedding endpoint
│   ├── [ollama-service][api_embed] - Native embedding
│   ├── [ollama-service][api_embeddings] - Native embeddings
│   └── [ollama-service][timing] - Performance metrics
│
├── app.ollama_client (FILE HANDLER)
│   ├── [ollama-service] -> upstream - Request start
│   ├── [ollama-service][timing] upstream - Response/error
│   └── [ollama-service] upstream - Success/failure
│
└── app.core.security (FILE HANDLER)
    ├── [ollama-service][security] - API key events
    └── [ollama-service][ratelimit] - Rate limit events
```

## Message Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Request                            │
│            POST /v1/chat with x-api-key header             │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
   ┌────▼─────────────────────┐      ┌───▼──────────────────────┐
   │ FastAPI Middleware       │      │ [DEBUG LOG 1]            │
   │ (Security dependency)    │      │ request received         │
   ├──────────────────────────┤      └──────────────────────────┘
   │ verify_api_key()         │
   │ - Check x-api-key header │      ┌──────────────────────────┐
   │ - Validate against KEY   │      │ [DEBUG LOG 2]            │
   │ - Return key or error    │      │ step=enforce_rate_limit  │
   └────┬──────────────────────┘      └──────────────────────────┘
        │
        │ (if valid, key returned)
        │
   ┌────▼──────────────────────────┐  ┌──────────────────────────┐
   │ Endpoint Handler               │  │ [DEBUG LOG 3]            │
   │ @app.post("/v1/chat")          │  │ step=rate_limit_ok       │
   ├────────────────────────────────┤  └──────────────────────────┘
   │ 1. Rate limit check            │
   │ 2. Input validation            │  ┌──────────────────────────┐
   │ 3. Payload construction        │  │ [DEBUG LOG 4-5]          │
   │ 4. Upstream call               │  │ validate_messages        │
   │ 5. Return response             │  │ validate_ok              │
   └────┬───────────────────────────┘  └──────────────────────────┘
        │
        │
   ┌────▼─────────────────────────────┐ ┌──────────────────────────┐
   │ post_ollama() in ollama_client.py │ │ [DEBUG LOG 6-7]          │
   ├───────────────────────────────────┤ │ construct_payload        │
   │ Make async HTTP POST to Ollama    │ │ payload_ready            │
   │ Handle errors and timeouts        │ └──────────────────────────┘
   │ Return response or raise error    │
   └────┬──────────────────────────────┘ ┌──────────────────────────┐
        │                                │ [INFO LOG 1]             │
        │ (network call to Ollama)      │ -> upstream POST /api/chat
        │                                └──────────────────────────┘
        │
        │
   ┌────▼──────────────────────┐      ┌──────────────────────────┐
   │ Ollama Service Response   │      │ [INFO LOG 2]             │
   │ (successful or error)     │      │ [timing] upstream ok     │
   └────┬─────────────────────┘       └──────────────────────────┘
        │
        │ (return to endpoint handler)
        │
   ┌────▼────────────────────────────┐ ┌──────────────────────────┐
   │ Response returned to client      │ │ [INFO LOG 3]             │
   │ (HTTP 200 + data or HTTP error)  │ │ response sent            │
   └────┬───────────────────────────────┘ └──────────────────────────┘
        │
        │
   ┌────▼──────────────────────┐
   │ Client receives response  │
   │ All logs written to file  │
   └───────────────────────────┘
```

## Summary

This logging architecture provides:

1. **Complete visibility** - Every step of request processing is logged
2. **Performance metrics** - Timing information at multiple levels
3. **Error tracking** - All errors logged with context
4. **Security audit** - All API key and rate limit events tracked
5. **Debugging support** - Detailed context for troubleshooting
6. **Minimal overhead** - Logging adds <1ms latency

The structured format makes logs easy to parse, analyze, and integrate with monitoring systems.
