# Operator / tool container: the security tools, in an isolated image on the labnet.
# Used for the fully-isolated variant (run tools via `docker exec lab-operator nmap web-target`).
# Tools install at BUILD time (needs Internet then); at RUN time the labnet can be `internal: true`.
#
# NOTE: seclists (545MB + ~2GB of tiny files) was dropped — on Docker-for-Windows its unpack is
# painfully slow and looks hung. A small curated web-content wordlist below is plenty for ffuf
# against the lab target; swap in seclists later if you want fuller coverage.
FROM kalilinux/kali-rolling

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      nmap \
      ffuf \
      masscan \
      curl \
      ca-certificates \
      dnsutils \
      iproute2 \
 && rm -rf /var/lib/apt/lists/*

# Small web-content wordlist for ffuf (no large download). Covers DVWA's discoverable paths.
RUN mkdir -p /usr/share/wordlists && printf '%s\n' \
      index.php login.php logout.php setup.php security.php instructions.php about.php \
      config config.inc.php admin administrator phpinfo.php robots.txt .htaccess \
      dvwa docs external vulnerabilities test backup uploads images css js includes \
      README.md server-status phpmyadmin index.html favicon.ico \
      > /usr/share/wordlists/common.txt

WORKDIR /work
