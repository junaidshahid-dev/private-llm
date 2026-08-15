# Operator / tool container: the security tools, in an isolated image on the labnet.
# Used for the fully-isolated variant (run tools via `docker exec lab-operator nmap web-target`).
# The tools are installed at BUILD time (needs Internet then); at RUN time the labnet can be made
# `internal: true` for zero outbound. seclists provides wordlists for ffuf.
FROM kalilinux/kali-rolling

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      nmap \
      ffuf \
      masscan \
      seclists \
      curl \
      ca-certificates \
      dnsutils \
      iproute2 \
 && rm -rf /var/lib/apt/lists/*

# common wordlist path used by ffuf_discover's default
RUN mkdir -p /usr/share/wordlists/dirb \
 && ln -sf /usr/share/seclists/Discovery/Web-Content/common.txt /usr/share/wordlists/dirb/common.txt \
    || true

WORKDIR /work
