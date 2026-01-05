output "resource_group_name" {
  value = azurerm_resource_group.rg.name
}

output "aml_workspace_name" {
  value = azurerm_machine_learning_workspace.aml.name
}

output "mlflow_tracking_uri" {
  value       = azurerm_machine_learning_workspace.aml.discovery_url
  description = "The MLflow tracking URI for the Azure ML Workspace"
}

output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}
