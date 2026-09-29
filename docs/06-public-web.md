# 06 - Public Web Access

## Simple public access

For a learning/portfolio VM:

```text
http://PUBLIC_IP:5000
```

MLflow UI.

```text
http://PUBLIC_IP:8000/docs
```

FastAPI Swagger.

## Public demo checklist (VM)

```bash
git clone https://github.com/ikhsannur1996/mlops.git && cd mlops
make up-public      # compose stack that accepts any Host header
sudo ufw allow 22/tcp && sudo ufw allow 5000,8000/tcp && sudo ufw enable
```

Open TCP 5000 and 8000 in the cloud security group as well (see
"Cloud firewall" below), then from your laptop:

```bash
curl http://PUBLIC_IP:8000/health
```

and browse `http://PUBLIC_IP:5000` for MLflow and
`http://PUBLIC_IP:8000/docs` for Swagger.

Without Docker: `MLFLOW_SERVER_ALLOWED_HOSTS='*' make mlflow-public` plus
`uvicorn app.main:app --host 0.0.0.0 --port 8000`.

## Allow the public host (MLflow)

MLflow validates the `Host` header to block DNS rebinding attacks. Its
default allow-list covers `localhost` and private network addresses only,
so opening `http://PUBLIC_IP:5000` fails with:

```text
Invalid Host header - possible DNS rebinding attack detected
```

Start MLflow with the public host allowed before you open the UI.

Local install:

```bash
MLFLOW_SERVER_ALLOWED_HOSTS="PUBLIC_IP" make mlflow
```

Docker — set it in `.env`, then restart:

```bash
MLFLOW_SERVER_ALLOWED_HOSTS=PUBLIC_IP
docker compose up -d
```

Use a comma-separated list for several hosts (e.g.
`PUBLIC_IP,mlflow.example.com`), or `*` to accept any host (demo only).
Port 8000 has no such check.

## No VM? Expose your laptop with a tunnel

A quick tunnel gives you a public HTTPS URL without touching your router
(the laptop must stay on). Use `*` for MLflow because the tunnel hostname
changes on every run.

```bash
# terminal 1 - MLflow accepting any Host header
MLFLOW_SERVER_ALLOWED_HOSTS='*' make mlflow

# terminal 2 - FastAPI on all interfaces
uvicorn app.main:app --host 0.0.0.0 --port 8000

# terminals 3 and 4 - one tunnel per port
cloudflared tunnel --url http://localhost:5000
cloudflared tunnel --url http://localhost:8000
```

Each tunnel prints an `https://...trycloudflare.com` URL;
`ngrok http 8000` works the same way if you prefer it.

Note: on macOS the AirPlay receiver already occupies port 5000 - disable it
first (see 07-troubleshooting.md).

## Verify listening ports

On the server:

```bash
sudo ss -lntp | grep -E ':5000|:8000'
```

Docker:

```bash
docker compose ps
```

## Test from the server

```bash
curl http://localhost:5000
curl http://localhost:8000/health
```

## Test from your laptop

```bash
curl http://PUBLIC_IP:8000/health
```

## Cloud firewall

The VM firewall and cloud security group are separate controls.

You may need both:

```text
Ubuntu UFW
+
Cloud firewall/security group
```

Open TCP 5000 and 8000 for the demo.

## CORS

FastAPI has no CORS middleware, which is fine for this demo: `/docs` and any
same-origin browser call work out of the box, and the simulator calls the
API from server side. If a browser app on another domain must call
`/predict`, add `CORSMiddleware` to `app/main.py` with the origins you
trust.

## Production recommendation

Do not expose MLflow and FastAPI directly on high ports in a real production environment.

Preferred:

```text
Internet
   |
 HTTPS 443
   |
Nginx / Caddy
   |
   +---- MLflow
   |
   +---- FastAPI
```

Use:

- HTTPS
- authentication
- restricted firewall rules
- secrets management
- backups
- private MLflow storage

The simple project intentionally keeps this out to avoid unnecessary complexity.
