# -----------------------------------------------------------------------------
# File: docker/qdrant/Dockerfile
# Purpose: Custom Qdrant image with curl for health checks
# Related: docker-compose.yml, health check configuration
# Tags: docker, qdrant, health-check, infrastructure
# -----------------------------------------------------------------------------

FROM qdrant/qdrant:v1.7.4

# Install curl for health checks and netcat for port checking
# Using apt-get with cleanup to minimize image size
RUN apt-get update && \
  apt-get install -y --no-install-recommends curl netcat-openbsd && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Verify installations
RUN curl --version && nc -h

# Keep the original entrypoint
ENTRYPOINT ["./entrypoint.sh"]
