# Newsies Helm Values Configuration Guide

## 🎯 Overview

This guide provides detailed configuration options for deploying Newsies 
microservices using Helm charts.

## 🏗️ Umbrella Chart Configuration

### Main Values File (`newsies/values.yaml`)

```yaml
# Global settings applied to all services
global:
  imageRegistry: ""
  imagePullSecrets: []
  storageClass: ""

# Infrastructure services
redis:
  enabled: true
  auth:
    enabled: true
    password: "your-secure-password"
  persistence:
    enabled: true
    size: 8Gi

chromadb:
  enabled: true
  persistence:
    enabled: true
    size: 20Gi
  resources:
    requests:
      memory: 2Gi
      cpu: 500m
    limits:
      memory: 4Gi
      cpu: 1000m

# Microservices
newsies-api:
  enabled: true
  replicaCount: 2
  ingress:
    enabled: true
    hosts:
      - host: newsies-api.example.com
        paths:
          - path: /
            pathType: Prefix

newsies-scraper:
  enabled: true
  replicaCount: 1
  resources:
    requests:
      memory: 1Gi
      cpu: 500m
    limits:
      memory: 2Gi
      cpu: 1000m

newsies-analyzer:
  enabled: true
  replicaCount: 1
  resources:
    requests:
      memory: 2Gi
      cpu: 1000m
    limits:
      memory: 4Gi
      cpu: 2000m

newsies-trainer:
  enabled: true
  replicaCount: 1
  resources:
    requests:
      memory: 4Gi
      cpu: 2000m
    limits:
      memory: 8Gi
      cpu: 4000m
      nvidia.com/gpu: 1  # Optional GPU support

newsies-cli:
  enabled: true
  replicaCount: 1
```

## 🔧 Service-Specific Configuration

### API Service Configuration

```yaml
# newsies-api/values.yaml
replicaCount: 2

image:
  repository: newsies-api
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: newsies.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: newsies-tls
      hosts:
        - newsies.example.com

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

# Environment-specific overrides
env:
  - name: ENVIRONMENT
    value: "production"
  - name: LOG_LEVEL
    value: "INFO"
  - name: API_WORKERS
    value: "4"
```

### Scraper Service Configuration

```yaml
# newsies-scraper/values.yaml
replicaCount: 1

image:
  repository: newsies-scraper
  pullPolicy: IfNotPresent
  tag: "latest"

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

# Scraper-specific environment variables
env:
  - name: SCRAPER_TIMEOUT
    value: "30"
  - name: MAX_CONCURRENT_REQUESTS
    value: "10"
  - name: USER_AGENT
    value: "Newsies-Bot/1.0"

# Node affinity for scraper workloads
nodeSelector:
  workload-type: "cpu-intensive"

# Tolerations for dedicated nodes
tolerations:
  - key: "scraper-node"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
```

### Analyzer Service Configuration

```yaml
# newsies-analyzer/values.yaml
replicaCount: 1

image:
  repository: newsies-analyzer
  pullPolicy: IfNotPresent
  tag: "latest"

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

# NLP-specific environment variables
env:
  - name: SPACY_MODEL
    value: "en_core_web_sm"
  - name: NLTK_DATA_PATH
    value: "/app/nltk_data"
  - name: MAX_TEXT_LENGTH
    value: "10000"

# Volume mounts for NLP models
volumeMounts:
  - name: nltk-data
    mountPath: /app/nltk_data
  - name: spacy-models
    mountPath: /app/spacy_models

volumes:
  - name: nltk-data
    persistentVolumeClaim:
      claimName: nltk-data-pvc
  - name: spacy-models
    persistentVolumeClaim:
      claimName: spacy-models-pvc
```

### Trainer Service Configuration

```yaml
# newsies-trainer/values.yaml
replicaCount: 1

image:
  repository: newsies-trainer
  pullPolicy: IfNotPresent
  tag: "latest"

resources:
  limits:
    cpu: 4000m
    memory: 8Gi
    nvidia.com/gpu: 1
  requests:
    cpu: 2000m
    memory: 4Gi
    nvidia.com/gpu: 1

# GPU node selection
nodeSelector:
  accelerator: nvidia-tesla-v100

tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"

# Training-specific environment variables
env:
  - name: CUDA_VISIBLE_DEVICES
    value: "0"
  - name: PYTORCH_CUDA_ALLOC_CONF
    value: "max_split_size_mb:512"
  - name: TRAINING_BATCH_SIZE
    value: "16"
  - name: LEARNING_RATE
    value: "0.0001"

# Persistent storage for models
persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: 100Gi
  mountPath: /app/models
```

## 🌍 Environment-Specific Configurations

### Development Environment

```yaml
# values-dev.yaml
global:
  imageRegistry: "localhost:5000"

redis:
  persistence:
    enabled: false

chromadb:
  persistence:
    enabled: false

newsies-api:
  replicaCount: 1
  ingress:
    enabled: false
  resources:
    requests:
      memory: 512Mi
      cpu: 250m
    limits:
      memory: 1Gi
      cpu: 500m

newsies-trainer:
  enabled: false  # Disable GPU service in dev
```

### Staging Environment

```yaml
# values-staging.yaml
global:
  imageRegistry: "registry.staging.example.com"

newsies-api:
  replicaCount: 1
  ingress:
    hosts:
      - host: newsies-staging.example.com

newsies-trainer:
  resources:
    limits:
      nvidia.com/gpu: 0  # No GPU in staging
```

### Production Environment

```yaml
# values-prod.yaml
global:
  imageRegistry: "registry.example.com"

redis:
  persistence:
    enabled: true
    size: 20Gi
    storageClass: "fast-ssd"
  resources:
    requests:
      memory: 2Gi
      cpu: 1000m
    limits:
      memory: 4Gi
      cpu: 2000m

chromadb:
  persistence:
    enabled: true
    size: 100Gi
    storageClass: "fast-ssd"

newsies-api:
  replicaCount: 3
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
  ingress:
    annotations:
      nginx.ingress.kubernetes.io/rate-limit: "100"
      nginx.ingress.kubernetes.io/ssl-redirect: "true"

newsies-scraper:
  replicaCount: 2
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10

newsies-analyzer:
  replicaCount: 2
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 8
```

## 🔒 Security Configuration

### Pod Security Standards

```yaml
# Security context for all services
podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000
```

### Network Policies

```yaml
# Network policy for API service
networkPolicy:
  enabled: true
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
    - from:
        - podSelector:
            matchLabels:
              app.kubernetes.io/name: newsies-scraper
      ports:
        - protocol: TCP
          port: 8000
```

## 📊 Monitoring Configuration

### Prometheus Integration

```yaml
# Monitoring configuration
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
    path: /metrics
    labels:
      app: newsies

# Grafana dashboard
grafana:
  dashboards:
    enabled: true
    annotations:
      grafana_folder: "Newsies"
```

## 🚀 Deployment Commands

### Install/Upgrade Commands

```bash
# Development deployment
helm install newsies ./newsies/ -f values-dev.yaml

# Staging deployment
helm upgrade --install newsies ./newsies/ -f values-staging.yaml

# Production deployment
helm upgrade --install newsies ./newsies/ -f values-prod.yaml

# Dry run to validate
helm install newsies ./newsies/ --dry-run --debug

# Template rendering
helm template newsies ./newsies/ -f values-prod.yaml
```

### Configuration Validation

```bash
# Validate Helm charts
helm lint ./newsies/

# Validate Kubernetes manifests
helm template newsies ./newsies/ | kubectl apply --dry-run=client -f -

# Check resource requirements
kubectl describe limitrange
kubectl describe resourcequota
```

---

## 📋 Best Practices

1. **Always use specific image tags** in production
2. **Set appropriate resource limits** to prevent resource starvation
3. **Enable persistence** for stateful services (Redis, ChromaDB)
4. **Configure monitoring** and alerting from day one
5. **Use secrets** for sensitive configuration data
6. **Test configurations** in staging before production deployment
7. **Implement proper backup strategies** for persistent data

This configuration guide provides the foundation for deploying Newsies in any 
Kubernetes environment with appropriate customization for your specific needs.
