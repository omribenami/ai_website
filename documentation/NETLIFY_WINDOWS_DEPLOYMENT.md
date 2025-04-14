# Step-by-Step Guide to Deploy AI Expert Guide Blog on Netlify (Windows)

This guide will walk you through the process of deploying your AI Expert Guide Blog to Netlify from a Windows environment, specifically overwriting your existing ai_website GitHub repository.

## Prerequisites

- Windows 10 or 11
- Git installed on your Windows computer ([Download Git for Windows](https://git-scm.com/download/win))
- Node.js and npm installed (version 16.x or higher recommended) ([Download Node.js](https://nodejs.org/))
- A GitHub account
- A Netlify account (free tier is sufficient)
- Your existing GitHub repository: https://github.com/omribenami/ai_website

## Step 1: Clone Your Existing Repository

First, let's clone your existing repository to your local machine:

```powershell
# Create a backup of your existing local directory if needed
ren D:\Downloads\ai_website D:\Downloads\ai_website_backup

# Clone your repository
git clone https://github.com/omribenami/ai_website.git D:\Downloads\ai_website
cd D:\Downloads\ai_website
```

## Step 2: Prepare the React Frontend

1. Make sure your React application is properly structured:

```powershell
# Navigate to your project directory
cd D:\Downloads\ai_website

# Create necessary directories if they don't exist
mkdir -p src\components
mkdir -p src\pages
mkdir -p public
```

2. Ensure your package.json has the correct dependencies and scripts:

```powershell
# Check if package.json exists and has the right content
notepad package.json
```

Your package.json should include:

```json
{
  "name": "ai-expert-guide-blog",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "@emotion/react": "^11.10.6",
    "@emotion/styled": "^11.10.6",
    "@mui/icons-material": "^5.11.16",
    "@mui/material": "^5.12.0",
    "axios": "^1.3.5",
    "chart.js": "^4.2.1",
    "react": "^18.2.0",
    "react-chartjs-2": "^5.2.0",
    "react-dom": "^18.2.0",
    "react-markdown": "^8.0.7",
    "react-router-dom": "^6.10.0",
    "react-scripts": "5.0.1",
    "react-syntax-highlighter": "^15.5.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "set NODE_OPTIONS=--openssl-legacy-provider && react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
```

3. Install dependencies:

```powershell
# Install dependencies
npm install
```

## Step 3: Configure Netlify Files

1. Create or update the netlify.toml file in your project root:

```powershell
# Create netlify.toml
notepad netlify.toml
```

Add the following content:

```toml
[build]
  command = "npm run build"
  publish = "build"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[build.environment]
  NODE_VERSION = "16.14.0"
  NPM_VERSION = "8.5.0"
```

2. Create the _redirects file in the public directory:

```powershell
# Create _redirects file
notepad public\_redirects
```

Add the following content:

```
/*    /index.html   200
```

## Step 4: Build Your Project

Now let's build your React application:

```powershell
# Build the project
npm run build
```

If you encounter any errors related to OpenSSL, make sure your build script in package.json includes the legacy provider flag:

```json
"build": "set NODE_OPTIONS=--openssl-legacy-provider && react-scripts build"
```

## Step 5: Deploy to Netlify

You have two options for deploying to Netlify:

### Option 1: Deploy via Netlify UI (Recommended for Windows Users)

1. Commit and push your changes to GitHub:

```powershell
# Add all files to git
git add .

# Commit changes
git commit -m "Prepare for Netlify deployment"

# Push to GitHub
git push origin master
```

2. Log in to your [Netlify account](https://app.netlify.com/)
3. Click "New site from Git"
4. Select GitHub as your Git provider
5. Authorize Netlify to access your GitHub account if prompted
6. Select your repository (ai_website)
7. Configure build settings:
   - Build command: `npm run build`
   - Publish directory: `build`
8. Click "Deploy site"

### Option 2: Deploy via Netlify CLI

1. Install Netlify CLI globally:

```powershell
npm install netlify-cli -g
```

2. Log in to Netlify:

```powershell
netlify login
```

3. Deploy your site:

```powershell
# Navigate to your project directory
cd D:\Downloads\ai_website

# Initialize Netlify (if not already done)
netlify init

# Deploy to production
netlify deploy --prod
```

When prompted for the publish directory, enter: `build`

## Step 6: Verify Your Deployment

1. After deployment completes, Netlify will provide a URL for your site
2. Visit the URL to ensure your site is working correctly
3. Test navigation and ensure all pages load properly

## Troubleshooting Common Windows Issues

### Issue: 'react-scripts' is not recognized

If you see an error like `'react-scripts' is not recognized as an internal or external command`:

```powershell
# Install react-scripts globally
npm install -g react-scripts

# Or install it locally
npm install react-scripts --save
```

### Issue: Node.js version compatibility

If you encounter Node.js version issues:

```powershell
# Install nvm-windows to manage Node.js versions
# Download from: https://github.com/coreybutler/nvm-windows/releases

# After installing nvm-windows, install and use Node.js 16
nvm install 16.14.0
nvm use 16.14.0
```

### Issue: OpenSSL errors

If you see errors related to OpenSSL:

```powershell
# Set the environment variable before building
set NODE_OPTIONS=--openssl-legacy-provider
npm run build
```

### Issue: Long path errors

Windows has path length limitations. To fix:

```powershell
# Enable long paths in Git
git config --system core.longpaths true

# Enable long paths in Windows (requires admin PowerShell)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## Maintaining Your Deployment

### Updating Your Site

To update your site after making changes:

```powershell
# Make your changes
# Then commit and push to GitHub
git add .
git commit -m "Update site content"
git push origin master
```

If you've set up continuous deployment, Netlify will automatically rebuild and deploy your site.

### Managing Environment Variables

To add environment variables:

1. Go to your site dashboard in Netlify
2. Navigate to Site settings > Build & deploy > Environment
3. Add your environment variables

## Conclusion

You've successfully deployed your AI Expert Guide Blog to Netlify from Windows! Your site is now accessible worldwide with Netlify's global CDN.

For more information, refer to:
- [Netlify Documentation](https://docs.netlify.com/)
- [React Deployment Guide](https://create-react-app.dev/docs/deployment/)
- [Netlify CLI Documentation](https://cli.netlify.com/)
