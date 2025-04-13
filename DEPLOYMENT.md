# AI Expert Guide Website Deployment Guide

This document provides instructions for deploying the AI Expert Guide e-learning platform.

## Prerequisites

- Docker and Docker Compose installed
- Node.js and npm (for local development)
- MongoDB (for local development without Docker)

## Project Structure

```
ai_website/
├── client/                  # Frontend React application
├── server/                  # Backend Node.js/Express application
├── content/                 # Course content
├── docker/                  # Docker configuration
│   ├── Dockerfile.client    # Client container configuration
│   └── Dockerfile.server    # Server container configuration
└── docker-compose.yml       # Multi-container setup
```

## Deployment Options

### 1. Docker Deployment (Recommended)

The easiest way to deploy the entire application stack is using Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd ai_website

# Start the application stack
docker-compose up -d

# The application will be available at:
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:5000
```

### 2. Static Website Deployment

For production deployment of the frontend as a static website:

```bash
# Build the React application
cd client
npm install
npm run build

# The built files will be in the 'build' directory
# These can be deployed to any static hosting service like:
# - Netlify
# - Vercel
# - AWS S3 + CloudFront
# - GitHub Pages
```

### 3. Separate Deployment

You can also deploy the frontend and backend separately:

#### Frontend:
```bash
cd client
npm install
npm start  # For development
npm run build  # For production build
```

#### Backend:
```bash
cd server
npm install
npm start  # Starts the server
```

## Environment Variables

### Client Environment Variables
Create a `.env` file in the client directory:

```
REACT_APP_API_URL=http://localhost:5000/api
```

### Server Environment Variables
Create a `.env` file in the server directory:

```
NODE_ENV=development
PORT=5000
MONGO_URI=mongodb://localhost:27017/ai_expert_guide
JWT_SECRET=your_jwt_secret_key_here
JWT_EXPIRE=30d
STRIPE_SECRET_KEY=your_stripe_secret_key_here
```

## Production Deployment Considerations

For a production deployment, consider the following:

1. **Security**:
   - Use HTTPS for all communications
   - Set secure and HTTP-only cookies
   - Implement rate limiting
   - Use environment variables for sensitive information

2. **Performance**:
   - Enable gzip compression
   - Implement caching strategies
   - Use a CDN for static assets
   - Optimize images and assets

3. **Scalability**:
   - Consider using a managed MongoDB service (MongoDB Atlas)
   - Deploy the backend to a scalable platform (AWS, Google Cloud, Azure)
   - Implement load balancing for the backend

4. **Monitoring**:
   - Set up logging and monitoring
   - Implement error tracking
   - Set up performance monitoring

## Continuous Integration/Continuous Deployment (CI/CD)

For automated deployments, consider setting up CI/CD pipelines using:
- GitHub Actions
- GitLab CI
- Jenkins
- CircleCI

## Backup and Recovery

Regularly backup your MongoDB database to prevent data loss.

## Support and Maintenance

For support and maintenance:
- Regularly update dependencies
- Monitor for security vulnerabilities
- Implement a staging environment for testing updates before production deployment

## Troubleshooting

If you encounter issues:

1. Check the logs:
   ```bash
   docker-compose logs
   ```

2. Verify environment variables are correctly set

3. Ensure all required ports are available

4. Check network connectivity between services
