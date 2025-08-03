# Newsies Microservices Testing Strategy

## 🎯 Testing Objectives

**Primary Goal**: Achieve 80%+ code coverage across all microservices before production deployment.

**Quality Gates**:
- Unit tests for all core business logic
- Integration tests for service interactions
- API contract tests for service boundaries
- End-to-end tests for critical user workflows

## 📦 Service-Specific Testing Requirements

### 1. newsies-common Package
**Coverage Target**: 90% (foundational utilities)

**Test Categories**:
- Redis task status coordination
- Visitor pattern implementations
- Data structures and utilities
- Migration utilities

**Key Test Files**:
- `tests/common/test_redis_task_status.py`
- `tests/common/test_visitor_pattern.py`
- `tests/common/test_data_structures.py`
- `tests/common/test_migration_utils.py`

### 2. newsies-clients Package
**Coverage Target**: 85% (database interactions)

**Test Categories**:
- Redis client operations
- ChromaDB client operations
- Session management
- Connection handling and retries

**Key Test Files**:
- `tests/clients/test_redis_client.py`
- `tests/clients/test_chromadb_client.py`
- `tests/clients/test_session_client.py`

### 3. newsies-api Package
**Coverage Target**: 80% (API endpoints)

**Test Categories**:
- FastAPI endpoint tests
- Authentication and authorization
- Request/response validation
- Background task execution
- Dashboard functionality

**Key Test Files**:
- `tests/api/test_endpoints.py`
- `tests/api/test_auth.py`
- `tests/api/test_background_tasks.py`
- `tests/api/test_dashboard.py`

### 4. newsies-scraper Package
**Coverage Target**: 80% (scraping logic)

**Test Categories**:
- News source scraping
- Article extraction
- Content parsing
- Error handling for network issues

**Key Test Files**:
- `tests/scraper/test_ap_news.py`
- `tests/scraper/test_article_extraction.py`
- `tests/scraper/test_content_parsing.py`
- `tests/scraper/test_pipeline.py`

### 5. newsies-analyzer Package
**Coverage Target**: 80% (NLP processing)

**Test Categories**:
- Text summarization
- Named entity recognition
- N-gram analysis
- Embedding generation

**Key Test Files**:
- `tests/analyzer/test_summarization.py`
- `tests/analyzer/test_ner.py`
- `tests/analyzer/test_ngrams.py`
- `tests/analyzer/test_embeddings.py`

### 6. newsies-trainer Package
**Coverage Target**: 75% (ML training)

**Test Categories**:
- Model training workflows
- LoRA adapter functionality
- GPU utilization (mocked)
- Model evaluation

**Key Test Files**:
- `tests/trainer/test_training_pipeline.py`
- `tests/trainer/test_lora_adapters.py`
- `tests/trainer/test_model_evaluation.py`

### 7. newsies-cli Package
**Coverage Target**: 70% (CLI interface)

**Test Categories**:
- Command parsing
- Interactive workflows
- Output formatting
- Error handling

**Key Test Files**:
- `tests/cli/test_commands.py`
- `tests/cli/test_interactive.py`
- `tests/cli/test_output.py`

## 🔧 Testing Infrastructure

### Testing Framework Stack
```python
# Core testing dependencies
pytest==7.4.0
pytest-cov==4.1.0
pytest-asyncio==0.21.0
pytest-mock==3.11.1
pytest-xdist==3.3.1  # Parallel test execution

# API testing
httpx==0.24.1
fastapi[testing]==0.100.0

# Database testing
fakeredis==2.16.0
pytest-docker==2.0.1

# Coverage reporting
coverage[toml]==7.2.7
```

### Test Configuration Files

#### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    --cov=newsies_common
    --cov=newsies_clients
    --cov=newsies_api
    --cov=newsies_scraper
    --cov=newsies_analyzer
    --cov=newsies_trainer
    --cov=newsies_cli
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    -v
    --tb=short
asyncio_mode = auto
```

#### coverage.toml
```toml
[tool.coverage.run]
source = [
    "newsies-common/newsies_common",
    "newsies-clients/newsies_clients", 
    "newsies-api/newsies_api",
    "newsies-scraper/newsies_scraper",
    "newsies-analyzer/newsies_analyzer",
    "newsies-trainer/newsies_trainer",
    "newsies-cli/newsies_cli"
]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
    "*/migrations/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:"
]
```

## 🧪 Test Categories and Patterns

### 1. Unit Tests
**Purpose**: Test individual functions and classes in isolation

```python
# Example: Redis task status unit test
import pytest
from unittest.mock import Mock, patch
from newsies_common.redis_task_status import RedisTaskStatus

def test_set_task_status():
    with patch('newsies_common.redis_task_status.redis_client') as mock_redis:
        task_status = RedisTaskStatus()
        task_status.set_status("task_123", "running")
        mock_redis.set.assert_called_once()
```

### 2. Integration Tests
**Purpose**: Test service interactions and database operations

```python
# Example: API integration test
import pytest
from fastapi.testclient import TestClient
from newsies_api.app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_get_articles_endpoint(client):
    response = client.post("/v1/run/get-news")
    assert response.status_code == 200
    assert "task_id" in response.json()
```

### 3. Contract Tests
**Purpose**: Ensure API contracts between services

```python
# Example: Service contract test
def test_scraper_api_contract():
    # Test that scraper service returns expected data structure
    response = scraper_client.get_articles()
    assert "articles" in response
    assert all("title" in article for article in response["articles"])
```

### 4. End-to-End Tests
**Purpose**: Test complete user workflows

```python
# Example: E2E workflow test
def test_complete_news_processing_workflow():
    # 1. Scrape articles
    # 2. Analyze content
    # 3. Generate summaries
    # 4. Store in database
    # 5. Query via API
    pass
```

## 🏗️ Test Environment Setup

### Docker Test Environment
```yaml
# docker-compose.test.yml
version: '3.8'
services:
  redis-test:
    image: redis:7-alpine
    ports:
      - "6380:6379"
  
  chromadb-test:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
```

### Test Data Management
- **Fixtures**: Reusable test data in `tests/fixtures/`
- **Factories**: Dynamic test data generation
- **Mocking**: External service dependencies

## 📊 Coverage Monitoring

### Coverage Targets by Package
- **newsies-common**: 90% (critical utilities)
- **newsies-clients**: 85% (database interactions)
- **newsies-api**: 80% (API endpoints)
- **newsies-scraper**: 80% (scraping logic)
- **newsies-analyzer**: 80% (NLP processing)
- **newsies-trainer**: 75% (ML training)
- **newsies-cli**: 70% (CLI interface)

### Quality Gates
- **Minimum overall coverage**: 80%
- **No untested critical paths**
- **All public APIs tested**
- **Error handling tested**

## 🚀 Testing Automation

### CI/CD Integration
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt
      - name: Run tests
        run: |
          pytest --cov-fail-under=80
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

## 📋 Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Set up testing infrastructure
- [ ] Create test configuration files
- [ ] Update existing tests for microservices
- [ ] Implement newsies-common tests (90% coverage)

### Phase 2: Core Services (Week 2)
- [ ] newsies-clients tests (85% coverage)
- [ ] newsies-api tests (80% coverage)
- [ ] Integration test framework

### Phase 3: Processing Services (Week 3)
- [ ] newsies-scraper tests (80% coverage)
- [ ] newsies-analyzer tests (80% coverage)
- [ ] Contract tests between services

### Phase 4: Specialized Services (Week 4)
- [ ] newsies-trainer tests (75% coverage)
- [ ] newsies-cli tests (70% coverage)
- [ ] End-to-end test suite

### Phase 5: Quality Assurance (Week 5)
- [ ] Performance tests
- [ ] Security tests
- [ ] Documentation and reporting
- [ ] CI/CD integration

## 🎯 Success Criteria

**Before Production Deployment**:
- ✅ 80%+ overall code coverage achieved
- ✅ All critical paths tested
- ✅ Integration tests passing
- ✅ Contract tests validating service boundaries
- ✅ End-to-end workflows tested
- ✅ CI/CD pipeline with automated testing
- ✅ Coverage reporting and monitoring

This comprehensive testing strategy ensures our microservices are robust, reliable, and ready for production deployment.
