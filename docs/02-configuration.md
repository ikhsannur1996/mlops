# 02 - Configuration

## Architecture

```text
Internet
   |
   +---- TCP 5000 ----> MLflow UI
   |
   +---- TCP 8000 ----> FastAPI
                           |
                           v
                       MLflow Model
                           |
                           v
                        SQLite
```

## Docker Compose

The main configuration is in:

```text
docker-compose.yml
```

Services:

- `mlflow`
- `api`

## Environment variables

FastAPI uses:

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
PREDICTION_DB=/app/predictions.db
```

For public access set `MLFLOW_SERVER_ALLOWED_HOSTS` so the MLflow UI
accepts your public host - see 06-public-web.md (`make up-public` is the
shortcut).

If running outside Docker:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## MLflow storage

This demo uses:

```text
mlflow.db
mlruns/
```

The SQLite database stores MLflow metadata.

The `mlruns/` directory stores artifacts.

For production, replace local storage with a proper database and object storage.

## Ports

Default:

```text
5000 = MLflow
8000 = FastAPI
```

Change the host-side ports in `docker-compose.yml` if necessary.

Example:

```yaml
ports:
  - "15000:5000"
```

Then MLflow is accessible at:

```text
http://PUBLIC_IP:15000
```

## Public IP

Find the public IP:

```bash
curl -4 ifconfig.me
```

Do not hardcode the public IP into the application. Docker binds the service to `0.0.0.0`.

## Firewall

If UFW is enabled:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 5000/tcp
sudo ufw allow 8000/tcp
sudo ufw status
```

Also open the same ports in the cloud provider firewall/security group.

For example:

```text
Inbound
TCP 22    SSH
TCP 5000  MLflow
TCP 8000  FastAPI
```

For production, prefer 80/443 through a reverse proxy and keep 5000/8000 private.
