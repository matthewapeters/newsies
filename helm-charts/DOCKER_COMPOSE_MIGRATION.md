# Docker Compose to Kubernetes Migration


This document outlines how the Docker Compose configuration from `docker/docker-compose.yml` and `scripts/newsies` has been migrated to Kubernetes using Helm charts.


## Original Docker Compose Configuration


### ChromaDB Service

 
- **Image**: `ghcr.io/chroma-core/chroma`
- **Port Mapping**: `8800:8000` (host:container)
- **Volume**: `${PWD}/chroma_data:/chroma/chroma`
- **Environment Variables**:
  - `CHROMA_SERVER_AUTHN_PROVIDER=chromadb.auth.basic_authn.BasicAuthenticationServerProvider`
  - `ANONYMIZED_TELEMETRY=TRUE`
  - `IS_PERSISTENT=TRUE`
  - `ALLOW_RESET=TRUE`
- **Authentication**: htpasswd-style credentials generated dynamically


### Redis Service

 
- **Image**: `redis`
- **Port Mapping**: `6379:6379`
- **Volume**: `${PWD}/redis_data:/data`
- **Command**: `redis-server --save 60 1 --loglevel warning --aclfile /usr/local/etc/redis/users.acl`
- **ACL Configuration**: Custom user authentication via `redis-users.acl`
- **Persistence**: Save every 60 seconds if at least 1 key changed


### Scripts/newsies Environment Variables

 
```bash
CHROMADB_HOST=localhost
CHROMADB_PORT=8800
CHROMADB_USER="user1"
CHROMADB_CREDS="${CHROMADB_USER}:${chromadb_password}"
REDIS_USER="user2"
REDIS_CREDS="${REDIS_USER}:${redis_password}"
TOKENIZERS_PARALLELISM=true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NLTK_DATA="~/nltk_data"
```


## Kubernetes Migration


### ChromaDB Helm Chart (`helm-charts/chromadb/`)

 
- **Deployment**: Matches Docker image and environment configuration
- **Service**: ClusterIP service on port 8000
- **Authentication**: Kubernetes Secret with htpasswd-style credentials
- **Persistence**: PersistentVolume using existing host path `/home/mpeters/Projects/newsies/docker/chroma_data`
- **Environment Variables**: All Docker Compose env vars preserved
- **Configuration**: `values.yaml` with authentication and persistence settings


### Redis Helm Chart (`helm-charts/redis/`)

 
- **Deployment**: Redis with ACL authentication
- **Service**: ClusterIP service on port 6379
- **Authentication**: Kubernetes Secret with ACL configuration
- **Persistence**: PersistentVolume using existing host path `/home/mpeters/Projects/newsies/docker/redis_data`
- **Command**: Exact match of Docker Compose Redis command
- **Configuration**: `values.yaml` with save settings and log level


### Umbrella Chart Updates (`helm-charts/newsies/`)

 
- **Dependencies**: Updated to use local Redis chart instead of Bitnami
- **Values**: Configured with Docker Compose credentials and settings
- **Volume Mapping**: Preserves existing volume mount points


## Key Migration Features


### 1. Volume Preservation

 
- **ChromaDB**: `/home/mpeters/Projects/newsies/docker/chroma_data` → PersistentVolume
- **Redis**: `/home/mpeters/Projects/newsies/docker/redis_data` → PersistentVolume
- **Mount Paths**: Preserved exactly as in Docker Compose


### 2. Authentication Migration

 
- **ChromaDB**: htpasswd-style credentials via Kubernetes Secrets
- **Redis**: ACL configuration via Kubernetes Secrets
- **Credentials**: Exact same usernames and passwords as Docker Compose


### 3. Network Configuration

 
- **Service Discovery**: Kubernetes Services replace Docker Compose networking
- **Port Mapping**: Internal ports preserved, external access via Services
- **DNS**: Kubernetes DNS replaces Docker Compose service names


### 4. Environment Variables

 
All environment variables from `scripts/newsies` are handled via:
- **ConfigMaps**: For non-sensitive configuration
- **Secrets**: For credentials and sensitive data
- **Deployment env**: For application-specific settings


## Deployment Commands


### Original Docker Compose

 
```bash
./scripts/newsies up    # Start services
./scripts/newsies down  # Stop services
```


### Kubernetes Equivalent

 
```bash
helm install newsies ./helm-charts/newsies/
helm uninstall newsies
```


## Verification

To verify the migration preserves all functionality:

1. **Data Persistence**: Existing data in `docker/chroma_data` and `docker/redis_data` will be accessible
2. **Authentication**: Same credentials work for both ChromaDB and Redis
3. **Configuration**: All environment variables and settings preserved
4. **Networking**: Services accessible via Kubernetes DNS names
5. **Volume Mounts**: Data persists across pod restarts

## Benefits of Kubernetes Migration

1. **Scalability**: Individual service scaling vs single Docker Compose stack
2. **Health Checks**: Built-in liveness and readiness probes
3. **Resource Management**: CPU/memory limits and requests
4. **Rolling Updates**: Zero-downtime deployments
5. **Service Discovery**: Native Kubernetes DNS and service mesh integration
6. **Monitoring**: Integration with Kubernetes monitoring stack
7. **Security**: RBAC, network policies, and pod security contexts
