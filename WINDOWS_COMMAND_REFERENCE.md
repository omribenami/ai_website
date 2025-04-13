# Windows Command Equivalents for Netlify Deployment

This reference guide provides Windows command equivalents for the commands used in the Netlify deployment process.

## Basic Command Equivalents

| Unix/Linux Command | Windows Command Prompt | Windows PowerShell |
|-------------------|------------------------|-------------------|
| `cd /path/to/dir` | `cd C:\path\to\dir` | `cd C:\path\to\dir` |
| `mkdir -p dir/subdir` | `mkdir dir\subdir` | `mkdir -Path dir\subdir` |
| `touch file.txt` | `type nul > file.txt` | `New-Item -Path file.txt -ItemType File` |
| `cat file.txt` | `type file.txt` | `Get-Content file.txt` |
| `ls` | `dir` | `Get-ChildItem` or `dir` |
| `rm file.txt` | `del file.txt` | `Remove-Item file.txt` |
| `rm -rf dir` | `rmdir /s /q dir` | `Remove-Item -Recurse -Force dir` |

## Git Commands (Same across platforms)

Git commands work the same way on Windows, but you might need to use Git Bash or adjust path separators:

```powershell
# Initialize repository
git init

# Add files
git add .

# Commit changes
git commit -m "Message"

# Add remote
git remote add origin https://github.com/username/repo.git

# Push to remote
git push -u origin main
```

## npm Commands (Same across platforms)

npm commands are the same on Windows, but you might encounter path issues:

```powershell
# Install dependencies
npm install

# Run scripts
npm run build

# Global installation
npm install -g package-name
```

## File Path Differences

| Unix/Linux | Windows |
|------------|---------|
| Forward slashes: `/` | Backslashes: `\` |
| No drive letters | Drive letters: `C:`, `D:` |
| Case-sensitive | Case-insensitive (usually) |

In PowerShell, you can often use forward slashes (`/`) instead of backslashes (`\`) and it will work correctly.

## Environment Variables

### Setting Environment Variables Temporarily

**Command Prompt:**
```cmd
set REACT_APP_API_URL=https://api.example.com
```

**PowerShell:**
```powershell
$env:REACT_APP_API_URL = "https://api.example.com"
```

### Setting Environment Variables Permanently

**Command Prompt:**
```cmd
setx REACT_APP_API_URL "https://api.example.com"
```

**PowerShell:**
```powershell
[Environment]::SetEnvironmentVariable("REACT_APP_API_URL", "https://api.example.com", "User")
```

## Creating Files with Content

### Creating a file with content in PowerShell:

```powershell
# Method 1: Using Set-Content
Set-Content -Path "netlify.toml" -Value @"
[build]
  command = "npm run build"
  publish = "build"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
"@

# Method 2: Using Out-File
@"
[build]
  command = "npm run build"
  publish = "build"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
"@ | Out-File -FilePath "netlify.toml"
```

### Creating a file with content in Command Prompt:

```cmd
(
echo [build]
echo   command = "npm run build"
echo   publish = "build"
echo.
echo [[redirects]]
echo   from = "/*"
echo   to = "/index.html"
echo   status = 200
) > netlify.toml
```

## Running Netlify CLI on Windows

Netlify CLI commands are the same on Windows:

```powershell
# Install Netlify CLI
npm install netlify-cli -g

# Login to Netlify
netlify login

# Initialize a new Netlify site
netlify init

# Deploy to Netlify
netlify deploy --prod
```

## Windows-Specific Tips

1. **Use PowerShell**: PowerShell is more powerful than Command Prompt and has better compatibility with Unix-like commands.

2. **Path Length Limitations**: Windows has a 260-character path length limit. Enable long paths in Git:
   ```powershell
   git config --system core.longpaths true
   ```

3. **Line Endings**: Configure Git to handle line endings properly:
   ```powershell
   git config --global core.autocrlf true
   ```

4. **Permission Issues**: Run PowerShell as Administrator for operations that require elevated privileges.

5. **Windows Terminal**: Consider using [Windows Terminal](https://github.com/microsoft/terminal) for a better command-line experience.

6. **WSL Option**: For a more Unix-like experience, consider using Windows Subsystem for Linux (WSL).
