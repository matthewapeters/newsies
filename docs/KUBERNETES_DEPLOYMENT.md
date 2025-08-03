# Newsies Kubernetes Deployment Guide

## 🎯 Overview

This guide covers deploying the Newsies Interactive News Explorer as a 
microservices architecture on Kubernetes using Helm charts.

## 🏗️ Architecture

Newsies has been migrated from a monolithic application to a cloud-native 
microservices architecture:

### Microservices
- **newsies-api** - FastAPI gateway service (REST API, dashboard)
- **newsies-scraper** - News scraping service (get-articles pipeline)
- **newsies-analyzer** - Content analysis service (analyze pipeline)
- **newsies-trainer** - Model training service (train-model pipeline, GPU-enabled)
- **newsies-cli** - Command-line interface service

### Infrastructure Services
- **Redis** - Distributed task coordination and caching
- **ChromaDB** - Vector database for embeddings and search

## 📦 Prerequisites

### Required Tools
- Kubernetes cluster (v1.20+)
- Helm 3.x
- kubectl configured for your cluster
- Docker (for image building)

### Required Resources
- **Minimum**: 4 CPU cores, 8GB RAM
- **Recommended**: 8 CPU cores, 16GB RAM
- **GPU**: Optional for trainer service (NVIDIA GPU with CUDA support)

## 🚀 Quick Start Deployment

### 1. Clone and Build Images
```bash
git clone https://github.com/matthewapeters/newsies
cd newsies

# Build all Docker images
./scripts/build-images.sh
```

### 2. Create Kubernetes Secrets
```bash
# Create Redis credentials
kubectl create secret generic redis-credentials \
  --from-literal=username=newsies \
  --from-literal=password=your-redis-password

# Create ChromaDB credentials  
kubectl create secret generic chromadb-credentials \
  --from-literal=username=newsies \
  --from-literal=password=your-chromadb-password
```

### 3. Deploy with Helm
```bash
# Add dependencies and deploy
cd helm-charts
helm dependency update newsies/
helm install newsies ./newsies/
```

### 4. Verify Deployment
```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/instance=newsies

# Check services
kubectl get services -l app.kubernetes.io/instance=newsies

# Access API service
kubectl port-forward service/newsies-api 8000:8000
```

## 🔧 Detailed Configuration

### Environment Variables

All services use these common environment variables:

```yaml
env:
  - name: REDIS_HOST
    value: "redis"
  - name: REDIS_PORT
    value: "6379"
  - name: REDIS_USER
    valueFrom:
      secretKeyRef:
        name: redis-credentials
        key: username
  - name: REDIS_PASSWORD
    valueFrom:
      secretKeyRef:
        name: redis-credentials
        key: password
  - name: CHROMADB_HOST
    value: "chromadb"
  - name: CHROMADB_PORT
    value: "8000"
  - name: CHROMADB_USER
    valueFrom:
      secretKeyRef:
        name: chromadb-credentials
        key: username
  - name: CHROMADB_PASSWORD
    valueFrom:
      secretKeyRef:
        name: chromadb-credentials
        key: password
```

### Service-Specific Configuration

#### API Service (newsies-api)
- **Port**: 8000
- **Replicas**: 2 (default)
- **Resources**: 1 CPU, 2GB RAM
- **Ingress**: Configurable for external access

#### Scraper Service (newsies-scraper)
- **Port**: 8000
- **Replicas**: 1 (default)
- **Resources**: 500m CPU, 1GB RAM
- **Dependencies**: Chrome/Chromium for web scraping

#### Analyzer Service (newsies-analyzer)
- **Port**: 8000
- **Replicas**: 1 (default)
- **Resources**: 2 CPU, 4GB RAM
- **Dependencies**: NLP libraries (spaCy, NLTK)

#### Trainer Service (newsies-trainer)
- **Port**: 8000
- **Replicas**: 1 (default)
- **Resources**: 4 CPU, 8GB RAM, GPU (optional)
- **Dependencies**: PyTorch, CUDA (for GPU)

#### CLI Service (newsies-cli)
- **Resources**: 100m CPU, 256MB RAM
- **Usage**: Interactive command-line interface

## 🔒 Security Configuration

### Pod Security Context
```yaml
podSecurityContext:
  fsGroup: 1000

securityContext:
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: false
  runAsNonRoot: true
  runAsUser: 1000
```

### Network Policies
All services communicate within the cluster network. External access is 
controlled via Ingress configuration.

## 📊 Monitoring and Health Checks

### Health Endpoints
All services expose health check endpoints:
- **Liveness Probe**: `/health` (HTTP GET)
- **Readiness Probe**: `/health` (HTTP GET)

### Monitoring Integration
Services are configured with Prometheus-compatible metrics endpoints for 
monitoring integration.

## 🔄 Scaling Configuration

### Horizontal Pod Autoscaling (HPA)
```yaml
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 5
  targetCPUUtilizationPercentage: 80
```

### Resource Requests and Limits
Each service has appropriate resource requests and limits configured to ensure 
optimal cluster resource utilization.

## 🚀 Advanced Deployment Options

### GPU Support for Trainer Service
For GPU-accelerated model training:

```yaml
# In newsies-trainer values.yaml
resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    nvidia.com/gpu: 1

nodeSelector:
  accelerator: nvidia-tesla-k80
```

### Persistent Storage
Configure persistent volumes for data persistence:

```yaml
persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: 100Gi
```

## 🔍 Troubleshooting

### Common Issues

1. **Pod Startup Failures**
   ```bash
   kubectl describe pod <pod-name>
   kubectl logs <pod-name>
   ```

2. **Service Connection Issues**
   ```bash
   kubectl get endpoints
   kubectl exec -it <pod-name> -- nslookup redis
   ```

3. **Resource Constraints**
   ```bash
   kubectl top pods
   kubectl describe nodes
   ```

### Debug Commands
```bash
# Check all resources
kubectl get all -l app.kubernetes.io/instance=newsies

# View logs for specific service
kubectl logs -l app.kubernetes.io/name=newsies-api

# Execute into pod for debugging
kubectl exec -it deployment/newsies-api -- /bin/bash
```

## 📈 Performance Tuning

### Resource Optimization
- Monitor CPU/memory usage with `kubectl top`
- Adjust resource requests/limits based on actual usage
- Use HPA for automatic scaling

### Database Optimization
- Configure Redis persistence for task coordination
- Optimize ChromaDB collection settings for vector search performance

## 🔄 Updates and Rollbacks

### Rolling Updates
```bash
# Update image version
helm upgrade newsies ./newsies/ --set image.tag=v1.1.0

# Rollback if needed
helm rollback newsies 1
```

### Zero-Downtime Deployments
All services are configured with rolling update strategies to ensure 
zero-downtime deployments.

## 📋 Maintenance

### Regular Tasks
- Monitor resource usage and adjust limits
- Update Docker images for security patches
- Backup persistent data (Redis, ChromaDB)
- Review and rotate secrets

### Backup Strategy
- Redis: Configure persistence and regular snapshots
- ChromaDB: Backup vector collections
- Application data: Export/import procedures

---

## 🎯 Next Steps

After successful deployment:
1. Configure monitoring and alerting
2. Set up CI/CD pipelines for automated deployments
3. Implement backup and disaster recovery procedures
4. Scale services based on usage patterns

For additional support, refer to the individual service documentation in each 
package directory.
