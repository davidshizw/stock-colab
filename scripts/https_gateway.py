"""Cloudflare Origin HTTPS Gateway (Port 8443 -> 127.0.0.1:8000) + UPnP & DDNS Keeper for stock.zs520.dev."""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import subprocess
import sys
from pathlib import Path

TLS_DIR = Path("/Users/davidshizw/Documents/GitHub/stock-colab/data_store/tls")
CERT_FILE = TLS_DIR / "origin.crt"
KEY_FILE = TLS_DIR / "origin.key"


def ensure_self_signed_cert() -> None:
    TLS_DIR.mkdir(parents=True, exist_ok=True)
    if CERT_FILE.exists() and KEY_FILE.exists():
        return
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(KEY_FILE),
            "-out",
            str(CERT_FILE),
            "-days",
            "3650",
            "-nodes",
            "-subj",
            "/CN=stock.zs520.dev",
        ],
        check=True,
        capture_output=True,
    )


def get_lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("192.168.1.1", 80))
        return s.getsockname()[0]


def sync_upnp_and_cloudflare() -> None:
    try:
        sys.path.insert(0, "/Users/davidshizw/Documents/GitHub/nas-playground/vpn")
        import gateway  # type: ignore

        lan_ip = get_lan_ip()
        control, typ = gateway.discover("192.168.1.1")
        wan_ip = gateway.soap(control, typ, "GetExternalIPAddress")["NewExternalIPAddress"]

        mappings = [
            (80, 8000, "STOCK-COLAB-HTTP"),
            (8080, 8000, "STOCK-COLAB-8080"),
            (8443, 8443, "STOCK-COLAB-HTTPS"),
        ]
        for ext_port, int_port, desc in mappings:
            args = {
                "NewRemoteHost": "",
                "NewExternalPort": ext_port,
                "NewProtocol": "TCP",
                "NewInternalPort": int_port,
                "NewInternalClient": lan_ip,
                "NewEnabled": 1,
                "NewPortMappingDescription": desc,
                "NewLeaseDuration": 0,
            }
            gateway.soap(control, typ, "AddPortMapping", args)

        gw = Path("/Users/davidshizw/Documents/GitHub/nas-playground/vpn/private/gateway")
        if (gw / "cloudflare.json").exists() and (gw / "cloudflare-token").exists():
            cf = json.loads((gw / "cloudflare.json").read_text())
            token = (gw / "cloudflare-token").read_text().strip()
            zone_id = cf["zone_id"]
            recs = gateway.cf_call(token, f"/zones/{zone_id}/dns_records?name=stock.zs520.dev")
            if recs:
                rec = recs[0]
                if rec.get("content") != wan_ip or not rec.get("proxied"):
                    gateway.cf_call(
                        token,
                        f"/zones/{zone_id}/dns_records/{rec['id']}",
                        "PATCH",
                        {"type": "A", "name": "stock.zs520.dev", "content": wan_ip, "proxied": True, "ttl": 1},
                    )
            else:
                gateway.cf_call(
                    token,
                    f"/zones/{zone_id}/dns_records",
                    "POST",
                    {"type": "A", "name": "stock.zs520.dev", "content": wan_ip, "proxied": True, "ttl": 1},
                )
        print(f"[stock.zs520.dev] Synced WAN={wan_ip} -> LAN={lan_ip} (80->8000, 8443->8443)", flush=True)
    except Exception as exc:
        print(f"[stock.zs520.dev] Sync warning: {exc}", flush=True)


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle_client(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
    try:
        backend_reader, backend_writer = await asyncio.open_connection("127.0.0.1", 8000)
    except Exception:
        client_writer.close()
        return
    await asyncio.gather(
        pipe(client_reader, backend_writer),
        pipe(backend_reader, client_writer),
    )


async def periodic_sync() -> None:
    while True:
        await asyncio.to_thread(sync_upnp_and_cloudflare)
        await asyncio.sleep(300)


async def main() -> None:
    ensure_self_signed_cert()
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(certfile=str(CERT_FILE), keyfile=str(KEY_FILE))

    server = await asyncio.start_server(handle_client, host="0.0.0.0", port=8443, ssl=ctx)
    asyncio.create_task(periodic_sync())
    print("HTTPS Origin Gateway listening on https://0.0.0.0:8443 -> http://127.0.0.1:8000", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
