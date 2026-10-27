# Deploying ContentBlitz to Azure Container Apps

This guide takes the Docker image from your laptop to a public HTTPS URL on Azure Container Apps (ACA). It assumes an Azure **free account** and macOS. Why the app runs as a single always-on replica: see [ADR 0003](adr/0003-single-replica-container-apps.md).

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
  --min-replicas 1 --max-replicas 1 \
  --cpu 1.0 --memory 2.0Gi \
  --secrets openai-api-key="$OPENAI_API_KEY" \
  --env-vars OPENAI_API_KEY=secretref:openai-api-key LANGSMITH_TRACING=false
```

`--registry-identity system` gives the app a managed identity with pull rights on your registry, so no registry password is stored. `--min-replicas 1 --max-replicas 1` keeps exactly one always-on replica (ADR 0003).

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

Open `https://<that-fqdn>` in a browser. A quick health check: `curl https://<fqdn>/_stcore/health` returns `ok`.

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
