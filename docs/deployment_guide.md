# Deploying ContentBlitz to Azure Container Apps

This guide takes the Docker image from your laptop to a public HTTPS URL on Azure Container Apps (ACA). It assumes an Azure **free account** and macOS. Why the app runs as a single replica: see [ADR 0003](adr/0003-single-replica-container-apps.md). While you're still building it scales to zero to save credit, and before the demo you switch it to always-on (step 9). Expected spend: see [Cost](#cost).

What you end up with:

```
Your Mac ──docker push──▶ Azure Container Registry ──pull──▶ Container App (1 replica, port 8501)
                                                                  │
                                                                  └── /data ◀── Azure Files share
```

## 0. Prerequisites

1. **Docker running locally.** Start Rancher Desktop (or Docker Desktop) and check with `docker info`.
2. **Azure CLI:**
   ```bash
   brew install azure-cli
   az login
   az extension add --name containerapp --upgrade
   ```
3. **Register the resource providers** (once per subscription, takes a minute or two):
   ```bash
   for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry Microsoft.Storage; do
     az provider register --namespace $ns --wait
   done
   ```

## 1. Set variables

Run these in the shell you'll use for every later step. Registry and storage names must be globally unique, lowercase letters and digits only, so change the suffix.

```bash
export RG=contentblitz-rg
export LOCATION=eastus
export ACR=contentblitzacr$RANDOM          # note the value: echo $ACR
export STORAGE=contentblitzst$RANDOM       # note the value: echo $STORAGE
export SHARE=contentblitz-data
export ENV_NAME=contentblitz-env
export APP=contentblitz
export TAG=v1
```

## 2. Create the resource group and container registry

```bash
az group create --name $RG --location $LOCATION
az acr create --resource-group $RG --name $ACR --sku Basic
```

ACR Basic costs roughly US$5/month, which comes out of your free credit.

## 3. Build and push the image

Azure runs `linux/amd64`. On an Apple Silicon Mac you **must** pass `--platform`, or the container will fail to start with an "exec format error".

```bash
az acr login --name $ACR
docker build --platform linux/amd64 -t $ACR.azurecr.io/$APP:$TAG .
docker push $ACR.azurecr.io/$APP:$TAG
```

(`az acr build` would build in the cloud instead, but ACR Tasks are often disabled on free-trial subscriptions. Local build + push always works.)

## 4. Create the Container Apps environment and storage

```bash
az containerapp env create --name $ENV_NAME --resource-group $RG --location $LOCATION

az storage account create --name $STORAGE --resource-group $RG --location $LOCATION \
  --sku Standard_LRS --kind StorageV2
export STORAGE_KEY=$(az storage account keys list -g $RG -n $STORAGE --query "[0].value" -o tsv)
az storage share-rm create --resource-group $RG --storage-account $STORAGE --name $SHARE --quota 5

az containerapp env storage set --name $ENV_NAME --resource-group $RG \
  --storage-name contentblitz-data \
  --azure-file-account-name $STORAGE --azure-file-account-key $STORAGE_KEY \
  --azure-file-share-name $SHARE --access-mode ReadWrite
```

## 5. Create the Container App

API keys are stored as Container Apps **secrets** and exposed to the app as environment variables. Set `OPENAI_API_KEY` in your shell first (`export OPENAI_API_KEY=sk-...`). Add more `--secrets` / `--env-vars` pairs the same way for SerpAPI, Tavily, LangSmith, etc.

```bash
az containerapp create --name $APP --resource-group $RG --environment $ENV_NAME \
  --image $ACR.azurecr.io/$APP:$TAG \
  --registry-server $ACR.azurecr.io --registry-identity system \
  --target-port 8501 --ingress external \
  --min-replicas 0 --max-replicas 1 \
  --cpu 0.5 --memory 1.0Gi \
  --secrets openai-api-key="$OPENAI_API_KEY" \
  --env-vars OPENAI_API_KEY=secretref:openai-api-key LANGSMITH_TRACING=false
```

`--registry-identity system` gives the app a managed identity with pull rights on your registry, so no registry password is stored. `--max-replicas 1` keeps it to a single replica (ADR 0003). `--min-replicas 0` lets it scale to zero when nobody is using it, which costs almost nothing. The trade-off is a 10–20 s cold start on the first visit, and in-memory conversations are lost when it scales down. That's fine while building; step 9 switches it to always-on for the demo. 0.5 vCPU / 1 GiB is the smallest size that should run Streamlit + LiteLLM. If the logs show the container being OOM-killed, go up to `--cpu 1.0 --memory 2.0Gi` (roughly double the compute cost).

## 6. Mount the Azure Files share at /data

The CLI can't add volumes directly, so export the app's YAML, edit it, and apply it:

```bash
az containerapp show --name $APP --resource-group $RG -o yaml > app.yaml
```

In `app.yaml`, under `properties.template`, add a `volumeMounts` entry to the container and a `volumes` section:

```yaml
    template:
      containers:
      - name: contentblitz
        image: ...                 # leave as is
        volumeMounts:              # add
        - volumeName: data
          mountPath: /data
      volumes:                     # add (same level as containers)
      - name: data
        storageName: contentblitz-data
        storageType: AzureFile
        mountOptions: uid=1000,gid=1000   # the image runs as user 1000
```

Then:

```bash
az containerapp update --name $APP --resource-group $RG --yaml app.yaml
rm app.yaml   # it contains secret references; don't commit it
```

## 7. Open it

```bash
az containerapp show --name $APP --resource-group $RG --query properties.configuration.ingress.fqdn -o tsv
```

Open `https://<that-fqdn>` in a browser. A quick health check: `curl https://<fqdn>/_stcore/health` returns `ok`. The first request after a quiet spell takes 10–20 s while the replica starts.

## 8. Set a budget alert

A $50/month budget on the resource group emails you at 50%, 80% and 100% of it. That's well above the expected spend (see [Cost](#cost)), so an alert means something is wrong, such as a crash-looping app or images piling up in the registry. Cost data lags by up to a day, so alerts aren't instant.

```bash
export ALERT_EMAIL=you@example.com
START=$(date -u +%Y-%m-01)
END=$(date -u -v1d -v+1y +%Y-%m-01)
az rest --method put \
  --url "https://management.azure.com/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RG/providers/Microsoft.Consumption/budgets/contentblitz-monthly?api-version=2021-10-01" \
  --body "{
    \"properties\": {
      \"category\": \"Cost\", \"amount\": 50, \"timeGrain\": \"Monthly\",
      \"timePeriod\": {\"startDate\": \"${START}T00:00:00Z\", \"endDate\": \"${END}T00:00:00Z\"},
      \"notifications\": {
        \"actual50\":  {\"enabled\": true, \"operator\": \"GreaterThanOrEqualTo\", \"threshold\": 50,  \"thresholdType\": \"Actual\", \"contactEmails\": [\"$ALERT_EMAIL\"]},
        \"actual80\":  {\"enabled\": true, \"operator\": \"GreaterThanOrEqualTo\", \"threshold\": 80,  \"thresholdType\": \"Actual\", \"contactEmails\": [\"$ALERT_EMAIL\"]},
        \"actual100\": {\"enabled\": true, \"operator\": \"GreaterThanOrEqualTo\", \"threshold\": 100, \"thresholdType\": \"Actual\", \"contactEmails\": [\"$ALERT_EMAIL\"]}
      }
    }
  }"
```

If this is refused on your subscription, create the same budget in the portal: **Cost Management** → **Budgets** → **Add**, scope `contentblitz-rg`.

## 9. Before the demo: switch to always-on

A few days before the demo or grading, stop the app scaling to zero, so there's no cold start and conversations survive (ADR 0003):

```bash
az containerapp update --name $APP --resource-group $RG --min-replicas 1
```

This raises the compute cost from about $0 to roughly $12/month. To go back afterwards, run the same command with `--min-replicas 0`. When grading is over, tear everything down (see the end of this guide).

## Deploying a new version

```bash
export TAG=v2
docker build --platform linux/amd64 -t $ACR.azurecr.io/$APP:$TAG .
docker push $ACR.azurecr.io/$APP:$TAG
az containerapp update --name $APP --resource-group $RG --image $ACR.azurecr.io/$APP:$TAG
```

## Continuous deployment from GitHub

Once the app exists (steps 1–7), every merge to `main` can deploy itself. [.github/workflows/deploy.yml](../.github/workflows/deploy.yml) runs after CI passes on `main`: it builds the image for `linux/amd64`, tags it with the commit SHA, pushes it to your registry, points the Container App at it, and waits for `/_stcore/health` on the new revision.

It logs in to Azure with OIDC, so no Azure password is stored in GitHub. Set that up once, in the shell where `$RG` and `$ACR` are still set, with the GitHub CLI logged in (`brew install gh && gh auth login`):

```bash
./scripts/setup_cd.sh
```

The script creates an app registration called `contentblitz-github-deploy` with a federated credential for this repo's `production` environment, and grants it `AcrPush` on the registry and `Contributor` on the resource group. It then sets the `AZURE_*` repository variables, creates the `production` environment (protected branches only), and protects `main`: changes only through PRs, and the CI jobs `test` and `docker` must pass. It is safe to re-run.

Until the script has run, the Deploy workflow skips itself. To deploy the current `main` without a merge: `gh workflow run Deploy`.

If your account can't create app registrations (`az ad app create` fails with *Insufficient privileges*), ask for the Application Developer role in Entra ID, or deploy manually as in the next section.

Adding or changing a secret:

```bash
az containerapp secret set --name $APP --resource-group $RG --secrets serpapi-api-key="$SERPAPI_API_KEY"
az containerapp update --name $APP --resource-group $RG --set-env-vars SERPAPI_API_KEY=secretref:serpapi-api-key
```

## Cost

Estimates for East US at late-2026 consumption rates, with the settings above. Check the [pricing calculator](https://azure.microsoft.com/pricing/calculator/) for current figures.

| Resource | How it's billed | Per month |
|---|---|---|
| Container App, `min-replicas 0` (building) | Billed only while running. Active: $0.000024/vCPU-s, $0.000003/GiB-s. The first 180k vCPU-s and 360k GiB-s each month are free (active time only). | ~$0–2 |
| Container App, `min-replicas 1` (demo) | Idle: $0.000003/vCPU-s and $0.000003/GiB-s, no free grant | ~$12 (worst case ~$34 if never idle) |
| Container Registry, Basic | $0.167/day, 10 GB storage included | ~$5 |
| Log Analytics (created with the environment) | First 5 GB ingested per month free | ~$0 |
| Azure Files, `/data` | Per GB stored | <$1 |
| Egress and requests | First 100 GB egress and 2M requests free | $0 |

**Total:** about $6/month while building, and $15–20/month once it's always-on.

LLM calls are billed by OpenAI, not Azure. They come out of your Azure credit only if you switch to Microsoft Foundry models.

**Free account rules.** The $200 credit is also the spending limit: at $200 Azure disables the subscription rather than charging your card. The credit also **expires 30 days after sign-up**, and the subscription is disabled then too. If you don't upgrade to pay-as-you-go within the following 30 days, deployed resources (including the `/data` share) are decommissioned. If grading runs past your credit's end date, upgrade before it. Remaining credit and the end date are shown under **Cost Management** in the portal.

**Registry clean-up.** Every CD deploy pushes a new SHA-tagged image. If the registry nears its 10 GB, keep the newest five and delete the rest:

```bash
az acr manifest list-metadata --registry $ACR --name $APP --orderby time_desc --query "[5:].digest" -o tsv \
  | xargs -I{} az acr repository delete --name $ACR --image "$APP@{}" --yes
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Revision fails, logs show `exec format error` | Image was built for arm64. Rebuild with `--platform linux/amd64`. |
| Image pull fails / `unauthorized` | Check the managed identity has `AcrPull`: `az role assignment list --assignee $(az containerapp show -n $APP -g $RG --query identity.principalId -o tsv)`. As a fallback: `az acr update -n $ACR --admin-enabled true` and recreate with `--registry-username`/`--registry-password`. |
| Page loads but stays on "Please wait..." | Websocket problem. Make sure `--max-replicas 1` and ingress `transport` is `auto`. |
| Permission denied writing to `/data` | The volume is missing `mountOptions: uid=1000,gid=1000`. |
| Anything else | Stream logs: `az containerapp logs show --name $APP --resource-group $RG --follow` |

## Using Microsoft Foundry models

Once you've deployed a model in a Foundry project (for example a `gpt-4o` deployment), add its key and endpoint as secrets/env vars (`AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`) and reference it in `config/models.yaml` as `azure/<deployment-name>`, e.g. `models: [azure/gpt-4o, openai/gpt-4o]`. Free-trial subscriptions often have little or no Azure OpenAI quota; check the Foundry portal's quota page first.

## Tearing it all down

```bash
az group delete --name $RG --yes --no-wait
```

This deletes everything above, including the registry and the stored data.
