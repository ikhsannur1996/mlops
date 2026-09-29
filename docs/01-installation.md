# 01 - Installation

## 1. Server requirements

Recommended minimum for a learning/portfolio VM:

- Ubuntu 22.04/24.04
- 2 vCPU
- 4 GB RAM
- 20 GB disk
- Public IPv4
- SSH access

The project can run on other Linux distributions, but commands below assume Ubuntu/Debian.

## 2. Update the server

```bash
sudo apt update
sudo apt upgrade -y
```

## 3. Install Git

```bash
sudo apt install -y git
git --version
```

## 4. Install Docker

```bash
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg   -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc]   https://download.docker.com/linux/ubuntu   $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable"   | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

docker --version
docker compose version
```

## 5. Allow Docker without sudo

```bash
sudo usermod -aG docker $USER
```

Log out and log in again, then verify:

```bash
docker ps
```

## 6. Copy the project

The repository is [`ikhsannur1996/mlops`](https://github.com/ikhsannur1996/mlops).

If using Git:

```bash
git clone https://github.com/ikhsannur1996/mlops.git
cd mlops
```

Or upload the ZIP and extract it:

```bash
unzip mlops.zip
cd mlops
```

## 7. Start the services

```bash
docker compose up -d --build
```

Check:

```bash
docker compose ps
docker compose logs -f
```

## 8. Open the browser

```text
http://PUBLIC_IP:5000
http://PUBLIC_IP:8000/docs
```

MLflow is on port 5000.

FastAPI Swagger is on port 8000.
