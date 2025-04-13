# Step-by-Step Guide to Deploy AI Expert Guide Website on Netlify (Windows)

This guide will walk you through the process of deploying the AI Expert Guide website frontend to Netlify specifically on Windows.

## Prerequisites

- Git installed on your Windows computer ([Download Git for Windows](https://git-scm.com/download/win))
- A GitHub account (or GitLab/Bitbucket)
- A Netlify account (free tier is sufficient)
- Node.js and npm installed on Windows ([Download Node.js](https://nodejs.org/en/download/))

## Step 1: Prepare the Frontend for Deployment

1. First, make sure the React frontend is ready for production:

```powershell
# Navigate to the client directory (use backslashes for Windows paths)
cd C:\path\to\ai_website\client

# Install dependencies if you haven't already
npm install

# Create a production build
npm run build
```

This will create a `build` folder with optimized production files.

## Step 2: Configure Environment Variables

1. Create a `.env` file in the client directory with your API endpoint:

```powershell
# Navigate to the client directory
cd C:\path\to\ai_website\client

# Create .env file using Notepad
notepad .env
```

2. Add the following to the .env file and save it:

```
REACT_APP_API_URL=https://your-backend-api-url.com/api
```

Note: For a complete deployment, you'll need to deploy the backend separately (e.g., on Heroku, AWS, etc.) and use that URL here.

## Step 3: Create Netlify Configuration Files

1. Create a `netlify.toml` file in the client directory:

```powershell
cd C:\path\to\ai_website\client
notepad netlify.toml
```

2. Add the following content to the `netlify.toml` file and save it:

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

```powershell
# First ensure the public directory exists
mkdir -p C:\path\to\ai_website\client\public

# Create the _redirects file
cd C:\path\to\ai_website\client\public
notepad _redirects
```

4. Add the following content to the `_redirects` file and save it:

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

2. Push your code to GitHub using Git Bash or PowerShell:

```powershell
# Navigate to your project root
cd C:\path\to\ai_website

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

Note: If you're using the default branch name "master" instead of "main", adjust the command accordingly.

## Step 5: Deploy to Netlify

### Option 1: Deploy via Netlify UI (Recommended for Windows Users)

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

1. Install Netlify CLI using PowerShell or Command Prompt:

```powershell
npm install netlify-cli -g
```

2. Log in to Netlify:

```powershell
netlify login
```

This will open a browser window for authentication.

3. Initialize Netlify in your project:

```powershell
cd C:\path\to\ai_website\client
netlify init
```

4. Follow the prompts:
   - Select "Create & configure a new site"
   - Choose your Netlify team
   - Provide a site name (or use the generated one)
   - Set your build command: `npm run build`
   - Set your publish directory: `build`

5. Deploy your site:

```powershell
netlify deploy --prod
```

## Step 6: Configure Custom Domain (Optional)

1. In the Netlify dashboard, go to your site
2. Click "Domain settings"
3. Click "Add custom domain"
4. Enter your domain name and follow the instructions to set up DNS

## Step 7: Set Up Continuous Deployment (Optional)

Netlify automatically sets up continuous deployment from your GitHub repository. Whenever you push changes to your repository, Netlify will automatically rebuild and redeploy your site.

## Windows-Specific Troubleshooting

### Issue: 'Command Not Found' Errors

- Ensure Node.js and Git are properly installed and added to your PATH
- Try restarting your command prompt or PowerShell after installation
- Use the full path to executables if needed (e.g., `C:\Program Files\nodejs\npm.cmd`)

### Issue: Path Separator Problems

- Windows uses backslashes (`\`) for file paths, but many commands expect forward slashes (`/`)
- In PowerShell, both types of slashes usually work, but if you encounter issues, try using forward slashes

### Issue: Permission Errors

- Try running Command Prompt or PowerShell as Administrator
- Check file permissions in your project directory

### Issue: Long Path Errors

- Windows has a default path length limitation
- Enable long path support in Git:
  ```powershell
  git config --system core.longpaths true
  ```

### Issue: Line Ending Differences

- Windows uses CRLF line endings while Unix systems use LF
- Configure Git to handle line endings:
  ```powershell
  git config --global core.autocrlf true
  ```

## Resources

- [Netlify Documentation](https://docs.netlify.com/)
- [Create React App Deployment Guide](https://create-react-app.dev/docs/deployment/#netlify)
- [Netlify CLI Documentation](https://docs.netlify.com/cli/get-started/)
- [Git for Windows](https://gitforwindows.org/)
- [Node.js for Windows](https://nodejs.org/en/download/)
