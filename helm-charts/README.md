# Newsies Kubernetes Helm Charts

This directory contains Helm charts for deploying the Newsies application as a microservices architecture on Kubernetes.

## Architecture Overview

The Newsies application is decomposed into the following components:

### Application Pods
- **newsies-api**: FastAPI gateway service with dashboard and session management
- **newsies-scraper**: News scraping and article processing pipeline
- **newsies-analyzer**: Content analysis, summarization, and graph generation
- **newsies-trainer**: LLM fine-tuning with LoRA adapters (GPU-enabled)

### Infrastructure Components
- **redis**: Session storage and task queuing
- **chromadb**: Vector embeddings and semantic search
- **postgresql**: Task status and metadata persistence

## Deployment

```bash
# Deploy infrastructure first
helm install newsies-redis ./redis
helm install newsies-chromadb ./chromadb
helm install newsies-postgres ./postgresql

# Deploy application services
helm install newsies-api ./newsies-api
helm install newsies-scraper ./newsies-scraper
helm install newsies-analyzer ./newsies-analyzer
helm install newsies-trainer ./newsies-trainer

# Or deploy everything with the umbrella chart
helm install newsies ./newsies
```

## Development

Each service chart includes:
- Deployment and Service manifests
- ConfigMaps for configuration
- Secrets for sensitive data
- PersistentVolumeClaims for data storage
- Horizontal Pod Autoscaler for scaling
- ServiceMonitor for Prometheus monitoring

## Volume Mapping

The charts leverage existing volume mappings for:
- ChromaDB data persistence
- Redis data persistence
- Model storage for training pipeline
- News article cache
