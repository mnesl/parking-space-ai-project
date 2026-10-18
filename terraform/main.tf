# ============================================================================
# Infrastructure as Code (Terraform) — AeroPark AI Reservation System
# Provisions:
#   1. Pinecone Serverless Vector Index (`parking-knowledge-base`)
#   2. Containerized Application Service configuration
# ============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    pinecone = {
      source  = "pinecone-io/pinecone"
      version = "~> 0.7.0"
    }
  }
}

variable "pinecone_api_key" {
  description = "API Key for Pinecone Serverless Vector Database"
  type        = string
  sensitive   = true
}

variable "embedding_dimension" {
  description = "Vector dimension (768 for Google Gemini text-embedding-004, 1536 for OpenAI text-embedding-3-small)"
  type        = number
  default     = 768
}

provider "pinecone" {
  api_key = var.pinecone_api_key
}

resource "pinecone_index" "parking_knowledge_base" {
  name      = "parking-knowledge-base"
  dimension = var.embedding_dimension
  metric    = "cosine"

  spec = {
    serverless = {
      cloud  = "aws"
      region = "us-east-1"
    }
  }
}

output "pinecone_index_host" {
  description = "Endpoint host of the provisioned Pinecone Serverless index"
  value       = pinecone_index.parking_knowledge_base.host
}

