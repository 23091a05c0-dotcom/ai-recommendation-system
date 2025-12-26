# GitHub Authentication Guide

## Problem
You're getting a **403 Permission Denied** error because GitHub no longer accepts password authentication over HTTPS. You need to use a **Personal Access Token (PAT)** instead.

## Solution: Use Personal Access Token

### Step 1: Create a Personal Access Token

1. Go to GitHub: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Give it a name: `AI Recommendation System Deployment`
4. Set expiration: Choose your preference (90 days recommended)
5. Select scopes:
   - ✅ **repo** (Full control of private repositories)
   - ✅ **workflow** (Update GitHub Action workflows)
6. Click **"Generate token"**
7. **COPY THE TOKEN** - you won't see it again!

### Step 2: Configure Git Credential Manager

#### Option A: Use Git Credential Manager (Recommended for Windows)

```bash
# This will prompt for username and token on next push
git config --global credential.helper manager-core
```

Then when you push:
```bash
git push -u origin main
```

When prompted:
- **Username**: `codersimha`
- **Password**: Paste your Personal Access Token (not your GitHub password!)

#### Option B: Store Token in URL (Quick but less secure)

```bash
# Remove existing remote
git remote remove origin

# Add remote with token in URL
git remote add origin https://YOUR_TOKEN@github.com/codersimha/ai-recommendation-system.git

# Push
git push -u origin main
```

Replace `YOUR_TOKEN` with your actual Personal Access Token.

#### Option C: Use SSH Instead (Most Secure)

If you have SSH keys set up:

```bash
# Remove HTTPS remote
git remote remove origin

# Add SSH remote
git remote add origin git@github.com:codersimha/ai-recommendation-system.git

# Push
git push -u origin main
```

### Step 3: Push to GitHub

After setting up authentication:

```bash
# Check current status
git status

# If you have uncommitted changes from the README edit
git add .
git commit -m "Update README"

# Push to GitHub
git push -u origin main
```

## Quick Fix (Try This First)

```bash
# This will prompt for credentials on next push
git config --global credential.helper store

# Now push - you'll be asked for username and token
git push -u origin main
```

When prompted:
- **Username**: `codersimha`
- **Password**: [Paste your Personal Access Token]

The credentials will be saved for future use.

## Verify Push Success

After successful push, verify on GitHub:
```
https://github.com/codersimha/ai-recommendation-system
```

You should see all 21 files including:
- README.md
- DEPLOYMENT.md
- render.yaml
- requirements.txt
- src/ directory
- models/ directory
- data/ directory

## Next Steps After Successful Push

1. Go to [Render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml`
5. Click "Create Web Service"
6. Wait 3-5 minutes for deployment

## Troubleshooting

**If you still get 403 error:**
- Make sure you're using the token, not your password
- Verify the token has `repo` scope
- Check token hasn't expired
- Ensure you copied the entire token

**If you get "remote origin already exists":**
```bash
git remote remove origin
git remote add origin https://github.com/codersimha/ai-recommendation-system.git
```

**To check what's configured:**
```bash
git remote -v
git config --list | grep credential
```
