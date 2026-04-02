# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
PII Compliance Auditor — Client.

Example:
    >>> with PIIEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     task = env.call_tool("get_task", task_id="task_1_easy")
    ...     result = env.call_tool("submit_findings",
    ...                            task_id="task_1_easy",
    ...                            findings_json='[{"text":"John","pii_type":"NAME"}]')
"""

from openenv.core.mcp_client import MCPToolClient


class PIIEnv(MCPToolClient):
    """
    Client for the PII Compliance Auditor environment.

    Inherits all functionality from MCPToolClient:
        list_tools()                    — discover available MCP tools
        call_tool(name, **kwargs)       — call a tool by name
        reset(**kwargs)                 — reset the environment
        step(action)                    — execute an action

    Available tools:
        list_tasks()                    — list all tasks
        get_task(task_id)               — get document + instructions
        submit_findings(task_id, findings_json) — submit and grade
        get_current_state()             — get episode state
    """
    pass  # MCPToolClient provides all needed functionality