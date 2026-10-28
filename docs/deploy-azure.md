# Fallback: deploy to Azure Container Apps (free monthly allowance)

Use this only if Render's free tier doesn't fit (for example, cold starts in a
live demo). The app is the same Docker image configured by environment
variables, so nothing in the code changes.

Azure Container Apps (Consumption plan) includes a monthly free allowance per
subscription (180,000 vCPU-seconds, 360,000 GiB-seconds, 2 million requests).
To keep the bill at zero, avoid what usually costs money:

| Avoid | Use instead |
|---|---|
| Azure Container Registry (paid) | GitHub Container Registry (`ghcr.io`, free) |
| Cosmos DB | MongoDB Atlas M0 (same `MONGO_URI`) |
| Uncapped Log Analytics | A workspace with a daily cap (e.g. 0.1 GB) |
| Always-on replicas | `--min-replicas 0` (scale to zero) |

Also create a **budget alert of $1** on the subscription.

## Steps

```bash
# 1. Build and push the image to GHCR (or let a GitHub Action do it)
docker build -f docker/Dockerfile -t ghcr.io/<user>/claims-assistant:latest .
docker push ghcr.io/<user>/claims-assistant:latest

# 2. Resource group + environment with a capped Log Analytics workspace
az group create -n claims-rg -l centralindia
az monitor log-analytics workspace create -g claims-rg -n claims-logs --quota 0.1
az containerapp env create -g claims-rg -n claims-env -l centralindia \
  --logs-workspace-id <workspace-customer-id> --logs-workspace-key <workspace-key>

# 3. The app: scale to zero, secrets as Container App secrets
az containerapp create -g claims-rg -n claims-app --environment claims-env \
  --image ghcr.io/<user>/claims-assistant:latest --target-port 8000 --ingress external \
  --min-replicas 0 --max-replicas 1 --cpu 0.5 --memory 1Gi \
  --secrets mongo-uri=<MONGO_URI> openai-key=<OPENAI_API_KEY> access-codes=<CODES> \
  --env-vars ENVIRONMENT=production MONGO_URI=secretref:mongo-uri \
             OPENAI_API_KEY=secretref:openai-key ACCESS_CODES=secretref:access-codes
```

If the GHCR package is private, add `--registry-server ghcr.io --registry-username <user>
--registry-password <PAT with read:packages>`. Point the health probe at
`/health/live`.
