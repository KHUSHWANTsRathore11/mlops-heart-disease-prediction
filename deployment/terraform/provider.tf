terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
  
  # Backend configuration for remote state
  # Values will be passed in via CLI in CI/CD pipeline
  backend "azurerm" {}
}

provider "azurerm" {
  features {}
}
