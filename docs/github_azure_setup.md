# GitHub to Azure Connectivity Guide

To allow GitHub Actions to deploy resources to your Azure Subscription (including Azure Student subscriptions), you need to create a **Service Principal** and store its credentials as GitHub Secrets.

## Prerequisites
*   Azure CLI installed locally or access to [Azure Cloud Shell](https://shell.azure.com).
*   Owner access to your Azure Subscription (default for Azure for Students).

## Step 1: Create a Service Principal

Run the following command to create a service principal with `Contributor` access to your entire subscription. This allows Terraform to create Resource Groups and resources.

```bash
# Get your Subscription ID
az account show --query id -o tsv

# Create the Service Principal
# Replace <SUBSCRIPTION_ID> with the id from above
# Replace <NAME> with a name like "github-actions-sp"
az ad sp create-for-rbac --name "github-actions-sp" --role Contributor --scopes /subscriptions/<SUBSCRIPTION_ID> --json-auth
```

### Output
The command will output a JSON object like this. **Save this information immediately; you cannot view the password again.**

```json
{
  "clientId": "00000000-0000-0000-0000-000000000000",
  "clientSecret": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "subscriptionId": "00000000-0000-0000-0000-000000000000",
  "tenantId": "00000000-0000-0000-0000-000000000000",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

## Step 2: Configure GitHub Secrets

1.  Go to your GitHub Repository.
2.  Navigate to **Settings** > **Secrets and variables** > **Actions**.
3.  Click **New repository secret**.
4.  Add the following secrets using the values from the JSON output above:

| Secret Name | Value from JSON |
| :--- | :--- |
| `AZURE_CLIENT_ID` | `clientId` |
| `AZURE_CLIENT_SECRET` | `clientSecret` |
| `AZURE_SUBSCRIPTION_ID` | `subscriptionId` |
| `AZURE_TENANT_ID` | `tenantId` |

## Why these permissions?
*   **Role: Contributor**: Terraform needs to create and modify resources (Clusters, Storage, etc.). `Contributor` is powerful enough to do this but cannot assign roles to others (unlike Owner).
*   **Scope: Subscription**: We start at the subscription level so Terraform can create the Resource Group itself. If you pre-created the Resource Group, you could scope this to just that group, but subscription scope is standard for IaC pipelines.
