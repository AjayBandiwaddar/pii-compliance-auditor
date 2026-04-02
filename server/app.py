# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
FastAPI application for the PII Compliance Auditor Environment.

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .pii_environment import PIIEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.pii_environment import PIIEnvironment

# Pass class (not instance) for WebSocket session isolation
app = create_app(
    PIIEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="pii_compliance_auditor",
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()