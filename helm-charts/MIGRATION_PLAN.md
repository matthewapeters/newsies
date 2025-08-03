# Newsies Kubernetes Migration Plan

## Overview
This document outlines the step-by-step plan for converting the Newsies monolithic application into a microservices architecture deployed on Kubernetes using Helm charts.

## Current State Analysis

### Identified Blockers
1. **Shared In-Memory State**: `TASK_STATUS` dictionary shared across all pipelines
2. **Threading Locks**: `RUN_LOCK` prevents horizontal scaling
3. **Tight Coupling**: Direct imports between pipeline modules
4. **Single Process**: All components run in same Python process

### Dependencies Mapping
- **Redis**: Session storage (already external)
- **ChromaDB**: Vector embeddings (already external) 
- **File System**: Article cache, model storage (needs PVC mapping)
- **Python Modules**: Shared utilities and data structures

## Migration Phases

### Phase 1: Decouple Shared State (Weeks 1-2)

#### 1.1 Replace TASK_STATUS with External Store
```python
# Current: In-memory dictionary
TASK_STATUS = AppStatus()

# Target: Redis-backed task tracking
class RedisTaskStatus:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def __setitem__(self, task_id, status):
        self.redis.hset(f"task:{task_id}", mapping=status)
        self.redis.expire(f"task:{task_id}", 86400)  # 24h TTL
```

#### 1.2 Remove Threading Locks
```python
# Current: Process-level lock
RUN_LOCK = threading.Lock()

# Target: Distributed task queue with Celery/RQ
from celery import Celery
app = Celery('newsies', broker='redis://newsies-redis:6379')

@app.task
def run_pipeline(pipeline_type, task_id, **kwargs):
    # Pipeline execution logic
```

#### 1.3 Create Pipeline Interfaces
```python
# Abstract base class for all pipelines
class PipelineService:
    def __init__(self, task_tracker, chromadb_client, redis_client):
        self.task_tracker = task_tracker
        self.chromadb = chromadb_client
        self.redis = redis_client
    
    async def execute(self, task_id: str, **kwargs):
        raise NotImplementedError
    
    async def health_check(self):
        return {"status": "healthy"}
```

### Phase 2: Extract Pipeline Services (Weeks 3-4)

#### 2.1 Create Scraper Service
```python
# newsies/services/scraper/main.py
from fastapi import FastAPI
from newsies.pipelines.get_articles import get_articles_pipeline

app = FastAPI()

@app.post("/v1/scrape")
async def scrape_articles(task_id: str):
    await get_articles_pipeline(task_id)
    return {"task_id": task_id, "status": "started"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

#### 2.2 Create Analyzer Service
```python
# newsies/services/analyzer/main.py
from fastapi import FastAPI
from newsies.pipelines.analyze import analyze_pipeline

app = FastAPI()

@app.post("/v1/analyze")
async def analyze_articles(task_id: str, archive: str = None):
    await analyze_pipeline(task_id, archive)
    return {"task_id": task_id, "status": "started"}
```

#### 2.3 Create Trainer Service
```python
# newsies/services/trainer/main.py
from fastapi import FastAPI
from newsies.pipelines.train_model import train_model_pipeline

app = FastAPI()

@app.post("/v1/train")
async def train_model(task_id: str):
    await train_model_pipeline(task_id)
    return {"task_id": task_id, "status": "started"}
```

### Phase 3: Containerize Services (Week 5)

#### 3.1 Create Dockerfiles
```dockerfile
# Dockerfile.api
FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY newsies/api/ ./api/
COPY newsies/session/ ./session/
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Dockerfile.scraper
FROM python:3.11-slim
WORKDIR /app
COPY requirements-scraper.txt .
RUN pip install -r requirements-scraper.txt
COPY newsies/services/scraper/ ./
COPY newsies/ap_news/ ./ap_news/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

# Dockerfile.trainer (GPU-enabled)
FROM nvidia/cuda:11.8-devel-ubuntu22.04
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements-trainer.txt .
RUN pip install -r requirements-trainer.txt
COPY newsies/services/trainer/ ./
COPY newsies/llm/ ./llm/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
```

#### 3.2 Split Requirements Files
```txt
# requirements-api.txt (lightweight)
fastapi==0.104.1
uvicorn==0.24.0
redis==5.0.1
pydantic==2.5.0

# requirements-scraper.txt (web scraping)
fastapi==0.104.1
beautifulsoup4==4.13.3
requests==2.31.0
sentence-transformers==2.2.2

# requirements-trainer.txt (ML/GPU)
torch==2.1.0
transformers==4.35.0
peft==0.6.0
datasets==2.14.0
```

### Phase 4: Deploy to Kubernetes (Week 6)

#### 4.1 Create Persistent Volumes
```yaml
# pv-articles.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: newsies-articles-pv
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: /data/newsies/articles
```

#### 4.2 Deploy Infrastructure
```bash
# Deploy Redis
helm install newsies-redis bitnami/redis \
  --set auth.password=newsies-redis-password \
  --set master.persistence.existingClaim=newsies-redis-data

# Deploy ChromaDB
helm install newsies-chromadb ./helm-charts/chromadb \
  --set persistence.existingClaim=newsies-chromadb-data

# Deploy PostgreSQL
helm install newsies-postgres bitnami/postgresql \
  --set auth.postgresPassword=newsies-postgres-password \
  --set auth.database=newsies
```

#### 4.3 Deploy Application Services
```bash
# Deploy services in dependency order
helm install newsies-api ./helm-charts/newsies-api
helm install newsies-scraper ./helm-charts/newsies-scraper
helm install newsies-analyzer ./helm-charts/newsies-analyzer
helm install newsies-trainer ./helm-charts/newsies-trainer
```

### Phase 5: Service Communication (Week 7)

#### 5.1 Update API Gateway
```python
# newsies/api/app.py - Updated for service calls
import httpx

@router_v1.get("/run/get-news")
async def run_get_news_pipeline(request: Request):
    task_id = str(uuid.uuid4())
    
    # Call scraper service
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://newsies-scraper:8001/v1/scrape",
            json={"task_id": task_id}
        )
    
    return {"task_id": task_id, "status": "queued"}
```

#### 5.2 Implement Health Checks
```python
# Add to each service
@app.get("/health")
async def health_check():
    # Check dependencies
    redis_ok = await check_redis_connection()
    chromadb_ok = await check_chromadb_connection()
    
    return {
        "status": "healthy" if all([redis_ok, chromadb_ok]) else "unhealthy",
        "dependencies": {
            "redis": redis_ok,
            "chromadb": chromadb_ok
        }
    }
```

### Phase 6: Monitoring & Observability (Week 8)

#### 6.1 Add Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, generate_latest

pipeline_executions = Counter('pipeline_executions_total', 'Total pipeline executions', ['pipeline_type'])
pipeline_duration = Histogram('pipeline_duration_seconds', 'Pipeline execution time', ['pipeline_type'])

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

#### 6.2 Distributed Tracing
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

tracer = trace.get_tracer(__name__)

@app.post("/v1/scrape")
async def scrape_articles(task_id: str):
    with tracer.start_as_current_span("scrape_articles") as span:
        span.set_attribute("task_id", task_id)
        await get_articles_pipeline(task_id)
```

## Testing Strategy

### Unit Testing
- Test each service independently
- Mock external dependencies (Redis, ChromaDB)
- Validate pipeline logic isolation

### Integration Testing
- Test service-to-service communication
- Validate data flow between components
- Test failure scenarios and recovery

### Load Testing
- Stress test individual services
- Validate autoscaling behavior
- Test resource limits and quotas

## Rollback Strategy

### Blue-Green Deployment
1. Deploy new services alongside existing monolith
2. Route small percentage of traffic to new services
3. Gradually increase traffic to new services
4. Keep monolith as fallback until fully validated

### Data Migration
1. Dual-write to both old and new storage systems
2. Validate data consistency
3. Switch reads to new system
4. Remove old storage system

## Success Metrics

### Performance
- Pipeline execution time: < 2x current performance
- Service startup time: < 30 seconds
- API response time: < 500ms p95

### Reliability
- Service uptime: > 99.9%
- Pipeline success rate: > 95%
- Recovery time: < 5 minutes

### Scalability
- Horizontal scaling: Support 10x current load
- Resource utilization: < 80% CPU/Memory
- Cost efficiency: < 1.5x current infrastructure cost

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| 1 | 2 weeks | Decouple shared state, remove locks |
| 2 | 2 weeks | Extract pipeline services |
| 3 | 1 week | Containerize all services |
| 4 | 1 week | Deploy to Kubernetes |
| 5 | 1 week | Service communication |
| 6 | 1 week | Monitoring & observability |

**Total: 8 weeks**

## Risk Mitigation

### Technical Risks
- **Data Loss**: Implement comprehensive backup strategy
- **Performance Degradation**: Extensive load testing before migration
- **Service Dependencies**: Circuit breakers and fallback mechanisms

### Operational Risks
- **Team Knowledge**: Cross-training on Kubernetes and microservices
- **Deployment Complexity**: Automated CI/CD pipelines
- **Monitoring Gaps**: Comprehensive observability from day one
