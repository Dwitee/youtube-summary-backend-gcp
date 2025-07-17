import os

# Redis connection string; fall back to localhost if not set
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")