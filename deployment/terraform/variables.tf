variable "location" {
  description = "The Azure region to deploy resources"
  type        = string
  default     = "southeastasia"
}

variable "prefix" {
  description = "A prefix used for all resources in this example"
  type        = string
  default     = "mlops-heart"
}

variable "subscription_id" {
    description = "The Azure subscription ID"
    type        = string
    sensitive   = true
    default = null
}

variable "tags" {
  description = "A mapping of tags to assign to the resources"
  type        = map(string)
  default = {
    Environment = "Development"
    Project     = "Heart-Disease-ML"
  }
}

variable "data_scientist_group_object_id" {
  description = "The Object ID of the Entra ID (Azure AD) Group for Data Scientists"
  type        = string
  # Default is null so we can check if it's provided before assigning
  default     = null 
}
