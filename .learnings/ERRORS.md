## [ERR-20260313-002] sessions_spawn-gateway-token

**Logged**: 2026-03-13T15:46:00Z
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
sessions_spawn failed due to gateway token mismatch (unauthorized).

### Error
```
gateway closed (1008): unauthorized: gateway token mismatch (set gateway.remote.token to match gateway.auth.token)
```

### Context
- Command/operation attempted: sessions_spawn via multi_tool_use.parallel
- Environment: local gateway ws://127.0.0.1:18789

### Suggested Fix
Align gateway.remote.token with gateway.auth.token in Clawdbot config.

### Metadata
- Reproducible: yes
- Related Files: /home/rootshell/.clawdbot/clawdbot.json

---
