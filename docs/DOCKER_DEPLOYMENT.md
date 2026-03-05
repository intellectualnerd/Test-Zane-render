# QueryGuard Docker Deployment Guide

## Prerequisites

- Docker and Docker Compose installed on your system
- At least 2GB of available RAM
- Port 8000 and 5432 available (or modify in docker-compose.yml)

## Quick Start

1. **Clone and navigate to the project directory:**
   ```bash
   cd QueryGuard-Backend
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   ```
   
3. **Edit the .env file and update the required values:**
   - `SECRET_KEY`: Change to a secure random string
   - `GOOGLE_API_KEY`: Add your Google API key if using AI features
   - Other optional configurations as needed

4. **Start the services:**
   ```bash
   docker-compose up -d
   ```

5. **Check service status:**
   ```bash
   docker-compose ps
   ```

6. **View logs:**
   ```bash
   # All services
   docker-compose logs -f
   
   # Specific service
   docker-compose logs -f backend
   docker-compose logs -f postgres
   ```

## Service URLs

- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **PostgreSQL**: localhost:5432

## Management Commands

### Stop services:
```bash
docker-compose down
```

### Stop and remove volumes (⚠️ This will delete all data):
```bash
docker-compose down -v
```

### Rebuild and restart:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### View real-time logs:
```bash
docker-compose logs -f
```

### Access database directly:
```bash
docker-compose exec postgres psql -U queryguard_user -d queryguard
```

### Execute commands in backend container:
```bash
docker-compose exec backend bash
```

## Data Persistence

The following data is persisted in Docker volumes:

- **PostgreSQL Data**: `postgres_data` volume
- **Chroma Vector Store**: `chroma_data` volume
- **Lineage Data**: Mapped to `./temp_lineage_data` directory

## Scaling and Production Considerations

### For Production Deployment:

1. **Security:**
   - Change all default passwords and secrets
   - Use proper SSL certificates
   - Set up firewall rules
   - Consider using Docker secrets for sensitive data

2. **Performance:**
   - Increase PostgreSQL memory settings
   - Configure proper backup strategy
   - Monitor resource usage

3. **High Availability:**
   - Use external managed PostgreSQL service
   - Set up load balancing for multiple backend instances
   - Implement proper logging and monitoring

### Environment Variables for Production:

Update these in your production `.env` file:

```env
# Use strong, unique secret key
SECRET_KEY=your-production-secret-key-256-bits-minimum

# Production database URL (consider managed service)
DATABASE_URL=postgresql+psycopg://user:password@prod-db:5432/queryguard

# Production Google API key
GOOGLE_API_KEY=your-production-google-api-key

# GitHub webhook secret for security
GITHUB_WEBHOOK_SECRET=your-production-github-webhook-secret
```

## Troubleshooting

### Common Issues:

1. **Port conflicts:**
   - Change ports in docker-compose.yml if 8000 or 5432 are already in use

2. **Permission errors:**
   - Ensure Docker has proper permissions
   - Check file ownership in mounted volumes

3. **Database connection errors:**
   - Wait for PostgreSQL to fully start before backend
   - Check health check status: `docker-compose ps`

4. **Backend fails to start:**
   - Check logs: `docker-compose logs backend`
   - Verify environment variables in .env file
   - Ensure all required dependencies are in requirements.txt

### Useful Commands for Debugging:

```bash
# Check if containers are healthy
docker-compose ps

# Check specific container logs
docker-compose logs backend
docker-compose logs postgres

# Test database connectivity
docker-compose exec postgres pg_isready -U queryguard_user

# Test backend health endpoint
curl http://localhost:8000/health

# Access backend container shell
docker-compose exec backend bash

# Check backend processes
docker-compose exec backend ps aux
```

## Development vs Production

This Docker Compose setup is suitable for:
- ✅ Development and testing
- ✅ Small production deployments
- ✅ Proof of concept

For large-scale production, consider:
- External managed database services
- Container orchestration (Kubernetes)
- Professional monitoring and logging solutions
- Load balancers and CDN
- Backup and disaster recovery strategies