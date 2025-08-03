#!/bin/bash

# Newsies Kubernetes Deployment Script
# This script automates the deployment of Newsies microservices to Kubernetes

set -e

# Configuration
NAMESPACE=${NAMESPACE:-newsies}
ENVIRONMENT=${ENVIRONMENT:-development}
HELM_RELEASE_NAME=${HELM_RELEASE_NAME:-newsies}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-""}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is installed (for image building)
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Function to create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace $NAMESPACE created"
    fi
}

# Function to create secrets
create_secrets() {
    log_info "Creating Kubernetes secrets..."
    
    # Redis credentials
    if kubectl get secret redis-credentials -n "$NAMESPACE" &> /dev/null; then
        log_warning "Redis credentials secret already exists"
    else
        read -s -p "Enter Redis password: " REDIS_PASSWORD
        echo
        kubectl create secret generic redis-credentials \
            --from-literal=username=newsies \
            --from-literal=password="$REDIS_PASSWORD" \
            -n "$NAMESPACE"
        log_success "Redis credentials secret created"
    fi
    
    # ChromaDB credentials
    if kubectl get secret chromadb-credentials -n "$NAMESPACE" &> /dev/null; then
        log_warning "ChromaDB credentials secret already exists"
    else
        read -s -p "Enter ChromaDB password: " CHROMADB_PASSWORD
        echo
        kubectl create secret generic chromadb-credentials \
            --from-literal=username=newsies \
            --from-literal=password="$CHROMADB_PASSWORD" \
            -n "$NAMESPACE"
        log_success "ChromaDB credentials secret created"
    fi
}

# Function to build Docker images
build_images() {
    log_info "Building Docker images..."
    
    if [[ -f "./scripts/build-images.sh" ]]; then
        chmod +x ./scripts/build-images.sh
        ./scripts/build-images.sh
        log_success "Docker images built successfully"
    else
        log_warning "Build script not found, assuming images are already built"
    fi
}

# Function to tag and push images (if registry is specified)
push_images() {
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        log_info "Tagging and pushing images to registry: $DOCKER_REGISTRY"
        
        services=("newsies-api" "newsies-scraper" "newsies-analyzer" "newsies-trainer" "newsies-cli")
        
        for service in "${services[@]}"; do
            log_info "Processing $service..."
            docker tag "$service:latest" "$DOCKER_REGISTRY/$service:latest"
            docker push "$DOCKER_REGISTRY/$service:latest"
        done
        
        log_success "All images pushed to registry"
    else
        log_info "No registry specified, skipping image push"
    fi
}

# Function to deploy with Helm
deploy_helm() {
    log_info "Deploying with Helm..."
    
    cd helm-charts
    
    # Update dependencies
    log_info "Updating Helm dependencies..."
    helm dependency update newsies/
    
    # Determine values file
    VALUES_FILE="values-${ENVIRONMENT}.yaml"
    if [[ ! -f "newsies/$VALUES_FILE" ]]; then
        log_warning "Environment-specific values file not found: $VALUES_FILE"
        log_info "Using default values.yaml"
        VALUES_FILE="values.yaml"
    fi
    
    # Deploy or upgrade
    log_info "Deploying Newsies with Helm..."
    helm upgrade --install "$HELM_RELEASE_NAME" ./newsies/ \
        --namespace "$NAMESPACE" \
        --values "newsies/$VALUES_FILE" \
        --timeout 10m \
        --wait
    
    log_success "Helm deployment completed"
    cd ..
}

# Function to verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Wait for pods to be ready
    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/instance="$HELM_RELEASE_NAME" \
        -n "$NAMESPACE" \
        --timeout=300s
    
    # Check pod status
    log_info "Pod status:"
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/instance="$HELM_RELEASE_NAME"
    
    # Check service status
    log_info "Service status:"
    kubectl get services -n "$NAMESPACE" -l app.kubernetes.io/instance="$HELM_RELEASE_NAME"
    
    # Check ingress (if enabled)
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        log_info "Ingress status:"
        kubectl get ingress -n "$NAMESPACE"
    fi
    
    log_success "Deployment verification completed"
}

# Function to show access information
show_access_info() {
    log_info "Access Information:"
    
    # API service access
    API_SERVICE=$(kubectl get service -n "$NAMESPACE" -l app.kubernetes.io/name=newsies-api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$API_SERVICE" ]]; then
        log_info "To access the API service locally:"
        echo "  kubectl port-forward service/$API_SERVICE 8000:8000 -n $NAMESPACE"
        echo "  Then visit: http://localhost:8000"
    fi
    
    # Ingress access
    INGRESS_HOST=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[0].spec.rules[0].host}' 2>/dev/null || echo "")
    if [[ -n "$INGRESS_HOST" ]]; then
        log_info "External access via Ingress:"
        echo "  https://$INGRESS_HOST"
    fi
}

# Function to clean up deployment
cleanup() {
    log_warning "Cleaning up deployment..."
    
    read -p "Are you sure you want to delete the deployment? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        helm uninstall "$HELM_RELEASE_NAME" -n "$NAMESPACE" || true
        kubectl delete namespace "$NAMESPACE" || true
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Main function
main() {
    echo "🚀 Newsies Kubernetes Deployment Script"
    echo "========================================"
    
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            create_namespace
            create_secrets
            build_images
            push_images
            deploy_helm
            verify_deployment
            show_access_info
            ;;
        "cleanup")
            cleanup
            ;;
        "verify")
            verify_deployment
            show_access_info
            ;;
        "build")
            build_images
            push_images
            ;;
        *)
            echo "Usage: $0 [deploy|cleanup|verify|build]"
            echo ""
            echo "Commands:"
            echo "  deploy  - Full deployment (default)"
            echo "  cleanup - Remove deployment"
            echo "  verify  - Verify existing deployment"
            echo "  build   - Build and push images only"
            echo ""
            echo "Environment Variables:"
            echo "  NAMESPACE           - Kubernetes namespace (default: newsies)"
            echo "  ENVIRONMENT         - Deployment environment (default: development)"
            echo "  HELM_RELEASE_NAME   - Helm release name (default: newsies)"
            echo "  DOCKER_REGISTRY     - Docker registry for images (optional)"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
