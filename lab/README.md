# Controlled security lab

A contained place to verify the **whole** security-tool chain — propose → approve → execute →
interpret → verify → report — against a deliberately-vulnerable target, without ever touching an
arbitrary external system. This is what moves `nmap`/`ffuf` from *locally tested* to *binary
verified* and *live-tested*.

```
lab/
├── docker-compose.yml       # isolated labnet: web-target (DVWA) + operator (tools)
├── operator/Dockerfile      # kali image with nmap/ffuf/masscan/seclists
├── targets/README.md        # the target(s) and how to reach/extend them
├── authorized_targets.yaml  # lab-only scope for the MCP gate (loopback/private only)
├── scripts/
│   ├── verify_lab.py        # pre-flight: connectivity, safety gate, authorization, binaries
│   ├── verify_lab_test.py   # proves the safety gate refuses public targets (no Docker needed)
│   └── reset_lab.py         # restore the target to a known state
└── README.md
```

> **I could not run any of this from where it was built** — no Docker, no tool binaries. The Python
> *logic* (the safety gate especially) is unit-tested; the compose/Dockerfile are correct-by-
> construction and get verified the first time **you** run them. Status stays honest: see the ladder.

## 1. Bring the lab up (on your machine)

```bash
docker compose -f lab/docker-compose.yml up -d --build
```

Then initialise DVWA once: open `http://127.0.0.1:8080/setup.php` → *Create / Reset Database*
(login `admin` / `password`). The target is published to **host loopback only** (`127.0.0.1:8080`),
never to your LAN or the Internet.

## 2. Authorize the lab (and nothing else)

Copy the entries from `lab/authorized_targets.yaml` into `configs/tools.yaml` under
`security_tools.authorized_targets`, and set `security_tools.enabled: true`. Target **class** never
implies authorization — this list does, and it contains only loopback/lab addresses.

## 3. Pre-flight

```bash
python lab/scripts/verify_lab.py 127.0.0.1:8080
```

It runs the sequence and refuses to go green unless: the target **resolves to the lab**
(loopback/private — a public address is hard-refused), the MCP config **authorizes** it, and the
port is reachable. It also reports which tools are installed.

## 4. First live test — nmap, then ffuf (before masscan)

Drive the real operator loop against the lab (nmap first — simplest to validate; then ffuf against
DVWA's PHP paths):

```bash
python serving/operator_loop.py "Recon the authorized lab target 127.0.0.1:8080: identify the service, then discover hidden web paths. One tool per step."
```

Expected chain: the model proposes `nmap_scan` on `127.0.0.1` → you approve → it runs and returns
real output → the model interprets it and proposes `ffuf_discover` → you approve → real results →
it correlates and produces a finding, with the verification verdict on each answer. Because the
tools actually run and return real output, the fabricated-output and phantom-action checks are
exercised for real.

## 5. Reset between runs

```bash
python lab/scripts/reset_lab.py            # restart target to baseline
python lab/scripts/reset_lab.py --recreate # full teardown + recreate
```

## Safety rules

- Lab is **loopback/labnet only**; for zero outbound, set `internal: true` on `labnet` and run
  tools inside the operator container (`docker exec lab-operator nmap web-target`).
- `verify_lab.py`'s gate **refuses any target that is not loopback/private/labnet** — tested.
- The MCP boundary is unchanged: the model proposes, **you** approve, only then does a tool run.

## Tool-readiness ladder

`implemented → gated → locally tested → binary verified → live-tested`

| tool | before the lab | after you run step 4 |
|---|---|---|
| nmap_scan | locally tested | binary verified → live-tested |
| ffuf_discover | locally tested | binary verified → live-tested |
| masscan_scan | locally tested | (validate last, after nmap/ffuf) |

Only after an actual run against the lab do `nmap`/`ffuf` reach **live-tested**. Don't mark them
sooner.
