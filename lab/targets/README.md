# Lab targets

The default target is **DVWA** (`vulnerables/web-dvwa`) defined in `../docker-compose.yml`. It is a
deliberately vulnerable PHP web app — good for the first live test because it has:

- an open web port for `nmap` to fingerprint, and
- classic discoverable paths for `ffuf` (`/login.php`, `/setup.php`, `/security.php`, `/instructions.php`, `/config/`).

## Reaching it

- **From the host** (the path the MCP tools use today): `http://127.0.0.1:8080/` — the container
  publishes port 80 to the host loopback only. Authorize `127.0.0.1`.
- **From the operator container** (isolated variant): `http://web-target/` at `172.28.0.10`.
  Authorize `web-target` / `172.28.0.10`.

## First DVWA setup

DVWA needs a one-time DB init: browse to `http://127.0.0.1:8080/setup.php` and click *Create /
Reset Database* (login `admin` / `password`). `reset_lab.py` restores this known state.

## Adding another target

Drop another service into `../docker-compose.yml` on `labnet` with a fixed `ipv4_address`, add its
address to `../authorized_targets.yaml`, and re-run `verify_lab.py`. Good alternatives:

- `bkimminich/juice-shop` (modern SPA; API/REST endpoints, port 3000)
- a minimal custom Flask/PHP target with a known-vulnerable endpoint

Keep every target on `labnet` and never publish beyond `127.0.0.1`.
