#!/bin/bash
# Build script for Newsies microservices Docker images

set -e

echo "🐳 Building Newsies microservices Docker images..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to build image with error handling
build_image() {
    local service=$1
    local dockerfile=$2
    local tag="newsies-${service}:latest"
    
    echo -e "${YELLOW}Building ${service} service...${NC}"
    
    if docker build -t "$tag" -f "$dockerfile" .; then
        echo -e "${GREEN}✅ Successfully built ${service} image${NC}"
    else
        echo -e "${RED}❌ Failed to build ${service} image${NC}"
        exit 1
    fi
}

# Build all service images
echo "📦 Building service images..."

build_image "api" "newsies-api/Dockerfile"
build_image "scraper" "newsies-scraper/Dockerfile" 
build_image "analyzer" "newsies-analyzer/Dockerfile"
build_image "trainer" "newsies-trainer/Dockerfile"
build_image "cli" "newsies-cli/Dockerfile"

echo -e "${GREEN}🎉 All images built successfully!${NC}"

# List built images
echo -e "${YELLOW}📋 Built images:${NC}"
docker images | grep newsies-

echo -e "${GREEN}✅ Build complete! Ready for deployment.${NC}"
