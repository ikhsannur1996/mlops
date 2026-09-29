# 07 - Troubleshooting

## Container status

```bash
docker compose ps
```

## Logs

MLflow:

```bash
docker compose logs -f mlflow
```

API:

```bash
docker compose logs -f api
```

## Restart

```bash
docker compose restart
```

Full rebuild:

```bash
docker compose down
docker compose up -d --build
```

## Port already in use

Check:

```bash
sudo ss -lntp | grep ':5000'
sudo ss -lntp | grep ':8000'
```

Change host ports in `docker-compose.yml` (and `scripts/start-mlflow.sh`
for a local MLflow), then point `MLFLOW_TRACKING_URI` at the new port.

On macOS the AirPlay receiver also listens on port 5000:

```bash
lsof -nP -i :5000
```

Disable it under System Settings > General > AirDrop & Handoff >
AirPlay Receiver, or move MLflow to another port.

## MLflow has no experiments

Run:

```bash
docker compose exec api python src/train.py
```

Then refresh MLflow.

## API says no registered model

Run:

```bash
docker compose exec api python src/train.py
```

Then test `/health` and `/predict`.

## Monitoring says not enough predictions

Make at least 10 calls to `/predict`.

## Invalid Host header - possible DNS rebinding attack detected

The MLflow UI rejected the `Host` header. The default allow-list covers
only localhost and private network addresses, so public IPs, domains,
`.local` names and VS Code forwarded ports are blocked.

Restart MLflow with your host allowed:

```bash
MLFLOW_SERVER_ALLOWED_HOSTS="PUBLIC_IP" make mlflow
```

For Docker, set `MLFLOW_SERVER_ALLOWED_HOSTS=PUBLIC_IP` in `.env` and run
`docker compose up -d` again. Use a comma-separated list for several hosts
or `*` to accept any (demo only). Port 8000 is not affected.

## Browser cannot access the VM

Check:

```bash
docker compose ps
sudo ufw status
sudo ss -lntp
```

Then check the cloud provider security group/firewall.

## Check public IP

```bash
curl -4 ifconfig.me
```

## Stop everything

```bash
docker compose down
```

Data is persisted through the project directory because the compose file mounts the project into the containers.
