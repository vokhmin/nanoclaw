FROM node:22-slim

# Install Docker CLI for Docker-out-of-Docker (socket is mounted from host)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y --no-install-recommends docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Mirror host path exactly so Docker-out-of-Docker mount paths are correct.
# process.cwd() inside container must match the host project path.
ENV HOME=/home/av
WORKDIR /home/av/projects/nanoclaw

CMD ["node", "dist/index.js"]
