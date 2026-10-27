#!/usr/bin/env bash
# One-time setup for continuous deployment (.github/workflows/deploy.yml).
#
# Run it after the manual deploy in docs/deployment_guide.md, in the same shell
# (it reads RG, ACR and APP), with `az login` and `gh auth login` done:
#
#   RG=contentblitz-rg ACR=contentblitzacr12345 ./scripts/setup_cd.sh
#
# It creates an Entra app registration that GitHub Actions logs in as via OIDC
# (no stored password), grants it push rights on the registry and Contributor on
# the resource group, sets the AZURE_* repo variables, creates the `production`
# environment, and protects main (PRs only, CI must pass). Safe to re-run.
set -euo pipefail

: "${RG:?set RG, e.g. RG=contentblitz-rg}"
: "${ACR:?set ACR to your registry name (echo \$ACR in the shell you deployed from)}"
APP=${APP:-contentblitz}
REPO=${REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}
SP_NAME=${SP_NAME:-contentblitz-github-deploy}

echo "==> Checking Azure resources ($RG / $ACR / $APP)"
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)
RG_ID=$(az group show --name "$RG" --query id -o tsv)
ACR_ID=$(az acr show --name "$ACR" --resource-group "$RG" --query id -o tsv)
az containerapp show --name "$APP" --resource-group "$RG" --query name -o tsv > /dev/null

echo "==> App registration $SP_NAME"
CLIENT_ID=$(az ad app list --display-name "$SP_NAME" --query "[0].appId" -o tsv)
if [ -z "$CLIENT_ID" ]; then
  CLIENT_ID=$(az ad app create --display-name "$SP_NAME" --query appId -o tsv)
fi
SP_OID=$(az ad sp show --id "$CLIENT_ID" --query id -o tsv 2> /dev/null \
  || az ad sp create --id "$CLIENT_ID" --query id -o tsv)

SUBJECT="repo:$REPO:environment:production"
if [ -z "$(az ad app federated-credential list --id "$CLIENT_ID" --query "[?subject=='$SUBJECT'].name" -o tsv)" ]; then
  az ad app federated-credential create --id "$CLIENT_ID" -o none --parameters "{
    \"name\": \"github-production\",
    \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"$SUBJECT\",
    \"audiences\": [\"api://AzureADTokenExchange\"]
  }"
fi

assign_role() {
  local role=$1 scope=$2
  if [ -z "$(az role assignment list --assignee "$SP_OID" --role "$role" --scope "$scope" --query "[0].id" -o tsv)" ]; then
    az role assignment create --assignee-object-id "$SP_OID" --assignee-principal-type ServicePrincipal \
      --role "$role" --scope "$scope" -o none
  fi
}
echo "==> Role assignments"
assign_role AcrPush "$ACR_ID"
assign_role Contributor "$RG_ID"

echo "==> GitHub environment and variables on $REPO"
# Only protected branches (main) may deploy to production.
gh api -X PUT "repos/$REPO/environments/production" --silent --input - << 'EOF'
{"deployment_branch_policy": {"protected_branches": true, "custom_branch_policies": false}}
EOF
gh variable set AZURE_CLIENT_ID --repo "$REPO" --body "$CLIENT_ID"
gh variable set AZURE_TENANT_ID --repo "$REPO" --body "$TENANT_ID"
gh variable set AZURE_SUBSCRIPTION_ID --repo "$REPO" --body "$SUBSCRIPTION_ID"
gh variable set AZURE_RG --repo "$REPO" --body "$RG"
gh variable set AZURE_ACR --repo "$REPO" --body "$ACR"
gh variable set AZURE_APP --repo "$REPO" --body "$APP"

echo "==> Protecting main (PRs only, CI jobs 'test' and 'docker' must pass)"
gh api -X PUT "repos/$REPO/branches/main/protection" --silent --input - << 'EOF'
{
  "required_status_checks": {"strict": false, "contexts": ["test", "docker"]},
  "enforce_admins": true,
  "required_pull_request_reviews": {"required_approving_review_count": 0},
  "restrictions": null
}
EOF

cat << EOF

Done. Every merge to main now deploys to $APP once CI passes.
To deploy the current main right away: gh workflow run Deploy --repo $REPO
Watch runs:                            gh run list --workflow Deploy --repo $REPO
EOF
