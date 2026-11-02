# -----------------------------------------------------------------------------
# File: tests/test_private_compose.py
# Purpose: Assert private compose does not publish PCS/data-store host ports
# Related: docker-compose.yml, docker-compose.prod.yml, docs/PRIVATE_STACK.md
# Tags: #test #private-network #infra-stack-up
# -----------------------------------------------------------------------------
"""Why: public hosts must keep the data plane unpublished.

Mutation: none. Input: compose YAML on disk. Output: pytest pass/fail.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SERVICES = ("postgres", "neo4j", "qdrant", "redis", "pcs")
FORBIDDEN_HOST_PORTS = {5432, 7474, 7687, 6333, 6334, 6379, 8000}


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def _published_host_ports(service: dict | None) -> set[int]:
    """Parse compose `ports` entries into host-side integers."""
    if not service:
        return set()
    found: set[int] = set()
    for item in service.get("ports") or []:
        text = str(item)
        # Forms: "5432:5432", "127.0.0.1:5432:5432", 5432
        if isinstance(item, int):
            found.add(item)
            continue
        left = text.split("/")[0]
        parts = left.split(":")
        if len(parts) == 1 and parts[0].isdigit():
            found.add(int(parts[0]))
        elif len(parts) >= 2 and parts[-2].isdigit():
            found.add(int(parts[-2]))
    return found


def test_base_compose_has_no_private_service_host_ports() -> None:
    base = _load_yaml("docker-compose.yml")
    services = base["services"]
    for name in PRIVATE_SERVICES:
        ports = _published_host_ports(services.get(name))
        assert not (ports & FORBIDDEN_HOST_PORTS), f"{name} publishes {ports}"


def test_prod_overlay_does_not_add_private_host_ports() -> None:
    prod = _load_yaml("docker-compose.prod.yml")
    services = prod.get("services") or {}
    for name in PRIVATE_SERVICES:
        ports = _published_host_ports(services.get(name))
        assert not ports, f"prod overlay must not publish {name}: {ports}"


def test_override_publishes_for_local_dev() -> None:
    override = _load_yaml("docker-compose.override.yml")
    services = override["services"]
    for name in PRIVATE_SERVICES:
        ports = _published_host_ports(services.get(name))
        assert ports, f"override should publish {name} for local digi"
