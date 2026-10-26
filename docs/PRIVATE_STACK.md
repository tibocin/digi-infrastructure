<!--
File: docs/PRIVATE_STACK.md
Purpose: Document private compose for PCS + data stores (unpublished host ports).
Related: docker-compose.yml, docker-compose.prod.yml, Makefile, tkt:infra-stack-up
Tags: #private-network #compose #infra-stack-up #security
-->

# Private stack (unpublished ports)

On a public host, PCS, Postgres, Neo4j, Qdrant, and Redis stay on the compose
network `digi-net` only. They are **not** published on the host.

## Documented command

```bash
cp config/env.example .env   # fill locally; never commit .env
make private-up              # docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
make private-health          # health via compose exec (no host localhost)
make assert-private-ports    # fail if data-plane ports are published
```

`make private-up` uses **explicit** `-f` files so `docker-compose.override.yml`
(local port publish) is **not** auto-merged.

## Local digi (ports published)

```bash
make up    # or make dev — loads override, publishes ports for laptop work
```

## Contract

| Service  | Private host publish | Reachability on digi-net   |
| -------- | -------------------- | -------------------------- |
| postgres | none                 | `postgres:5432`            |
| neo4j    | none                 | `neo4j:7687` / `:7474`     |
| qdrant   | none                 | `qdrant:6333`              |
| redis    | none                 | `redis:6379`               |
| pcs      | none                 | `http://pcs:8000`          |

Do not record secret values or origin addresses in git, health logs committed
to the repo, or runbooks that land in this ticket.
