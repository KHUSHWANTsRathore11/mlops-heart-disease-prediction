data "azurerm_client_config" "current" {}

resource "azurerm_resource_group" "rg" {
  name     = "${var.prefix}-rg"
  location = var.location
  tags     = var.tags
}

# Storage Account (Required for AML)
resource "azurerm_storage_account" "storage" {
  name                     = replace("${var.prefix}stor${random_string.suffix.result}", "-", "")
  location                 = azurerm_resource_group.rg.location
  resource_group_name      = azurerm_resource_group.rg.name
  account_tier             = "Standard"
  account_replication_type = "LRS"
  tags                     = var.tags
}

# Key Vault (Required for AML)
resource "azurerm_key_vault" "kv" {
  name                     = "${var.prefix}-kv-${random_string.suffix.result}"
  location                 = azurerm_resource_group.rg.location
  resource_group_name      = azurerm_resource_group.rg.name
  tenant_id                = data.azurerm_client_config.current.tenant_id
  sku_name                 = "standard"
  purge_protection_enabled = false
  tags                     = var.tags
}

# Application Insights (Required for AML)
resource "azurerm_application_insights" "app_insights" {
  name                = "${var.prefix}-ai"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  application_type    = "web"
  tags                = var.tags
}

# Container Registry (Required for AML Environments)
resource "azurerm_container_registry" "acr" {
  name                = replace("${var.prefix}acr${random_string.suffix.result}", "-", "")
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = var.tags
}

# Azure Machine Learning Workspace
resource "azurerm_machine_learning_workspace" "aml" {
  name                    = "${var.prefix}-aml-ws"
  location                = azurerm_resource_group.rg.location
  resource_group_name     = azurerm_resource_group.rg.name
  application_insights_id = azurerm_application_insights.app_insights.id
  key_vault_id            = azurerm_key_vault.kv.id
  storage_account_id      = azurerm_storage_account.storage.id
  container_registry_id   = azurerm_container_registry.acr.id
  
  public_network_access_enabled = true
  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# Random suffix to ensure unique names for storage/kv/acr
resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

# Role Assignment: Grant "AzureML Data Scientist" to the Entra Group
# Only creates this if the variable is provided
resource "azurerm_role_assignment" "data_scientist_role" {
  count                = var.data_scientist_group_object_id != null ? 1 : 0
  scope                = azurerm_machine_learning_workspace.aml.id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = var.data_scientist_group_object_id
}
