# AI Expert Guide Blog Deployment Guide

This document provides comprehensive instructions for deploying the AI Expert Guide Blog, a personal blog platform with gated content access, payment integration with Buy Me A Coffee, and admin capabilities.

## System Architecture

The AI Expert Guide Blog consists of three main components:

1. **Frontend Client**: React-based single-page application
2. **Backend API**: Node.js/Express server
3. **Database**: MongoDB for data storage

All components are containerized using Docker for easy deployment.

## Prerequisites

Before deployment, ensure you have the following installed:

- Docker and Docker Compose
- Git
- Node.js and npm (for local development only)

## Deployment Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-expert-guide-blog.git
cd ai-expert-guide-blog
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory based on the provided `env.example`:

```bash
cp env.example .env
```

Edit the `.env` file and update the following variables:

- `MONGO_USERNAME` and `MONGO_PASSWORD`: Set secure credentials for MongoDB
- `JWT_SECRET`: Set a strong secret key for JWT authentication
- `SMTP_*`: Configure email settings for password reset functionality
- `API_URL` and `CLIENT_URL`: Set to your domain or IP address

### 3. Update Nginx Configuration (if needed)

If you're deploying to a custom domain, update the `client/nginx.conf` file with your domain name.

### 4. Build and Start the Containers

```bash
docker-compose up -d
```

This command will:
- Build the Docker images for the client and server
- Start the MongoDB database
- Start the backend API server
- Start the frontend client with Nginx

### 5. Initialize Admin User

After deployment, you need to create an admin user. Use the following API endpoint:

```bash
curl -X POST http://your-domain/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","email":"your-email@example.com","password":"your-secure-password","role":"admin"}'
```

### 6. Verify Deployment

Access the following URLs to verify your deployment:

- Frontend: http://your-domain
- Backend API: http://your-domain/api/v1/content (should return a list of modules)

## Content Management

### Adding Course Content

1. Log in as an admin
2. Navigate to the Admin Dashboard
3. Use the Content Management section to add new modules and lessons

### Managing Users

1. Log in as an admin
2. Navigate to the Admin Dashboard
3. Use the User Management section to view, edit, or delete users

### Verifying Payments

1. Log in as an admin
2. Navigate to the Admin Dashboard
3. Use the Payment Verification section to approve or reject payment submissions

## Maintenance

### Backing Up the Database

```bash
docker exec -it mongodb mongodump --out /data/backup
docker cp mongodb:/data/backup ./backup
```

### Updating the Application

```bash
git pull
docker-compose down
docker-compose up -d --build
```

### Viewing Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs api
docker-compose logs client
docker-compose logs mongodb
```

## Troubleshooting

### Common Issues

1. **Connection refused to MongoDB**:
   - Check MongoDB container is running: `docker ps`
   - Verify MongoDB credentials in `.env` file

2. **API endpoints returning 500 errors**:
   - Check API logs: `docker-compose logs api`
   - Verify MongoDB connection in API logs

3. **Frontend not loading**:
   - Check Nginx logs: `docker-compose logs client`
   - Verify API URL in `.env` file

4. **Payment verification not working**:
   - Check email configuration in `.env` file
   - Verify Buy Me A Coffee integration settings

## Security Considerations

1. Always use HTTPS in production
2. Regularly update dependencies
3. Implement rate limiting for API endpoints
4. Use strong passwords for all accounts
5. Regularly backup your database

## Support

For additional support, please contact the developer at your-email@example.com.
