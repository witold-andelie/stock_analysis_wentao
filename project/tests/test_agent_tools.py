import json
from pathlib import Path

from driftrev import agent_tools
from driftrev.api import app


def test_tools_json_valid_and_current():
    path = Path(__file__).resolve().parents[1] / "agent" / "tools.json"
    tools = json.loads(path.read_text())
    assert tools == agent_tools.TOOLS
    for t in tools:
        f = t["function"]
        assert t["type"] == "function" and f["description"]
        assert f["parameters"]["type"] == "object"
        assert set(f["parameters"]["required"]) <= set(f["parameters"]["properties"])


def test_every_tool_maps_to_route():
    paths = app.openapi()["paths"]
    for name, r in agent_tools.ROUTES.items():
        op = paths[r["path"]][r["method"].lower()]
        assert op["operationId"] == name
