# Step-by-Step Guide to Deploy AI Expert Guide Website on Netlify

This guide will walk you through the process of deploying the AI Expert Guide website frontend to Netlify.

## Prerequisites

- Git installed on your computer
- A GitHub account (or GitLab/Bitbucket)
- A Netlify account (free tier is sufficient)
- Node.js and npm installed locally

## Step 1: Prepare the Frontend for Deployment

1. First, make sure the React frontend is ready for production:

```bash
# Navigate to the client directory
cd /path/to/ai_website/client

# Install dependencies if you haven't already
npm install

# Create a production build
npm run build
```

This will create a `build` folder with optimized production files.

## Step 2: Configure Environment Variables

1. Create a `.env` file in the client directory with your API endpoint:

```
REACT_APP_API_URL=https://your-backend-api-url.com/api
```

Note: For a complete deployment, you'll need to deploy the backend separately (e.g., on Heroku, AWS, etc.) and use that URL here.

## Step 3: Create Netlify Configuration Files

1. Create a `netlify.toml` file in the client directory:

```bash
cd /path/to/ai_website/client
touch netlify.toml
```

2. Add the following content to the `netlify.toml` file:

```toml
[build]
  command = "npm run build"
  publish = "build"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

This configuration tells Netlify how to build your site and sets up redirects for client-side routing.

3. Create a `_redirects` file in the `public` directory:

```bash
cd /path/to/ai_website/client/public
touch _redirects
```

4. Add the following content to the `_redirects` file:

```
/*    /index.html   200
```

This ensures that all routes are handled by your React application.

## Step 4: Push Your Code to GitHub

1. Create a new GitHub repository:
   - Go to [GitHub](https://github.com)
   - Click the "+" icon in the top right and select "New repository"
   - Name your repository (e.g., "ai-expert-guide")
   - Make it public or private as needed
   - Click "Create repository"

2. Push your code to GitHub:

```bash
# Navigate to your project root
cd /path/to/ai_website

# Initialize Git repository if not already done
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit for Netlify deployment"

# Add GitHub repository as remote
git remote add origin https://github.com/your-username/ai-expert-guide.git

# Push to GitHub
git push -u origin main
```

## Step 5: Deploy to Netlify

### Option 1: Deploy via Netlify UI

1. Log in to your [Netlify account](https://app.netlify.com/)
2. Click "New site from Git"
3. Select GitHub as your Git provider
4. Authorize Netlify to access your GitHub account if prompted
5. Select your repository (ai-expert-guide)
6. Configure build settings:
   - Build command: `npm run build`
   - Publish directory: `build`
7. Click "Show advanced" and add your environment variables:
   - Key: `REACT_APP_API_URL`
   - Value: `https://your-backend-api-url.com/api`
8. Click "Deploy site"

### Option 2: Deploy via Netlify CLI

1. Install Netlify CLI:

```bash
npm install netlify-cli -g
```

2. Log in to Netlify:

```bash
netlify login
```

3. Initialize Netlify in your project:

```bash
cd /path/to/ai_website/client
netlify init
```

4. Follow the prompts:
   - Select "Create & configure a new site"
   - Choose your Netlify team
   - Provide a site name (or use the generated one)
   - Set your build command: `npm run build`
   - Set your publish directory: `build`

5. Deploy your site:

```bash
netlify deploy --prod
```

## Step 6: Configure Custom Domain (Optional)

1. In the Netlify dashboard, go to your site
2. Click "Domain settings"
3. Click "Add custom domain"
4. Enter your domain name and follow the instructions to set up DNS

## Step 7: Set Up Continuous Deployment (Optional)

Netlify automatically sets up continuous deployment from your GitHub repository. Whenever you push changes to your repository, Netlify will automatically rebuild and redeploy your site.

## Troubleshooting

### Issue: Build Fails

- Check your build logs in the Netlify dashboard
- Ensure all dependencies are correctly listed in package.json
- Verify that your build command works locally

### Issue: API Calls Not Working

- Confirm your backend API is deployed and accessible
- Check that you've set the correct REACT_APP_API_URL environment variable
- Ensure your backend has CORS configured to allow requests from your Netlify domain

### Issue: Routes Not Working

- Verify that the `_redirects` file is in the `public` directory
- Check that your `netlify.toml` file has the correct redirects configuration

## Next Steps

1. Set up your backend API (if not already done)
2. Configure authentication services
3. Set up a database for your application
4. Connect payment processing services

## Resources

- [Netlify Documentation](https://docs.netlify.com/)
- [Create React App Deployment Guide](https://create-react-app.dev/docs/deployment/#netlify)
- [Netlify CLI Documentation](https://docs.netlify.com/cli/get-started/)
