#!/bin/bash

# ============================================================
# 🚀 College RAG Chatbot — Quick Git Upload Script
# ============================================================
# This script automatically sets up and uploads your project to GitHub
# Usage: bash upload_to_github.sh
# ============================================================

set -e  # Exit on error

PROJECT_DIR="/Users/yogesh/c0DE/bproject pratice/sentiment/chhhh"
REPO_NAME="college-rag-chatbot"

echo "════════════════════════════════════════════════════════"
echo "🚀 College RAG Chatbot — GitHub Upload"
echo "════════════════════════════════════════════════════════"
echo ""

# Step 0: Navigate to project
echo "📍 Navigating to project directory..."
cd "$PROJECT_DIR"
echo "✅ Current directory: $(pwd)"
echo ""

# Step 1: Check if git is installed
echo "📍 Checking if git is installed..."
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install from https://git-scm.com/download/mac"
    exit 1
fi
echo "✅ Git is installed: $(git --version)"
echo ""

# Step 2: Initialize git (if not already)
echo "📍 Setting up git repository..."
if [ -d .git ]; then
    echo "✅ Git repository already exists"
else
    git init
    echo "✅ Initialized git repository"
fi
echo ""

# Step 3: Create .gitignore
echo "📍 Setting up .gitignore..."
cat > .gitignore << 'EOF'
# Virtual Environment
.venv/
venv/
ENV/
env/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Cache & Temp
.DS_Store
*.pkl
*.cache
.streamlit/

# Environment variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Embeddings cache (regeneratable)
embeddings/embeddings_cache.pkl
EOF
echo "✅ Created .gitignore"
echo ""

# Step 4: Stage all files
echo "📍 Staging all files..."
git add .
echo "✅ Files staged"
echo ""

# Step 5: Create initial commit
echo "📍 Creating initial commit..."
if git diff-index --quiet HEAD --; then
    echo "ℹ️  No changes to commit (files already committed)"
else
    git commit -m "Initial commit: College RAG Chatbot with Streamlit UI and RAG pipeline"
    echo "✅ Initial commit created"
fi
echo ""

# Step 6: Set main as default branch
echo "📍 Setting main as default branch..."
if git rev-parse --verify main > /dev/null 2>&1; then
    echo "✅ Main branch already set"
else
    git branch -M main
    echo "✅ Main branch configured"
fi
echo ""

# Step 7: Get GitHub URL
echo "════════════════════════════════════════════════════════"
echo "📝 GITHUB REPOSITORY URL NEEDED"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Before continuing, create a repository on GitHub:"
echo "  1. Go to https://github.com/new"
echo "  2. Enter name: $REPO_NAME"
echo "  3. Select 'Public' visibility"
echo "  4. DO NOT initialize with README, .gitignore, or license"
echo "  5. Click 'Create repository'"
echo ""
echo "Then copy the repository URL (HTTPS format preferred):"
echo "  Example: https://github.com/YOUR_USERNAME/$REPO_NAME.git"
echo ""

read -p "📌 Enter your GitHub repository URL: " GITHUB_URL

if [ -z "$GITHUB_URL" ]; then
    echo "❌ No URL provided. Exiting."
    exit 1
fi

echo ""
echo "📍 Adding remote origin..."

# Remove existing remote if present
if git remote get-url origin > /dev/null 2>&1; then
    echo "ℹ️  Remote already exists. Removing..."
    git remote remove origin
fi

git remote add origin "$GITHUB_URL"
echo "✅ Remote added: $GITHUB_URL"
echo ""

# Step 8: Push to GitHub
echo "════════════════════════════════════════════════════════"
echo "📤 READY TO PUSH"
echo "════════════════════════════════════════════════════════"
echo ""
echo "⚠️  IMPORTANT: Prepare for authentication prompt"
echo ""
echo "If using HTTPS (recommended for first time):"
echo "  • GitHub username: your GitHub username"
echo "  • Password: your GitHub Personal Access Token"
echo "    - Get one at: https://github.com/settings/tokens"
echo "    - Scopes needed: ✓ repo"
echo ""
echo "If using SSH:"
echo "  • Make sure your SSH key is added to GitHub"
echo "  • https://github.com/settings/keys"
echo ""

read -p "📍 Ready to push to GitHub? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📍 Pushing to GitHub (main branch)..."
    git push -u origin main

    if [ $? -eq 0 ]; then
        echo ""
        echo "════════════════════════════════════════════════════════"
        echo "✅ SUCCESS! Your project is now on GitHub!"
        echo "════════════════════════════════════════════════════════"
        echo ""
        echo "🎉 View your repository at:"
        echo "   $GITHUB_URL"
        echo ""
        echo "📋 Next steps:"
        echo "   1. Go to your GitHub repo page"
        echo "   2. Add a description and topics"
        echo "   3. Share the link with others!"
        echo ""
    else
        echo "❌ Push failed. Check your credentials and try again."
        exit 1
    fi
else
    echo "⚠️  Push cancelled. You can push later with:"
    echo "   git push -u origin main"
    exit 0
fi

echo "════════════════════════════════════════════════════════"
echo "🚀 Setup complete!"
echo "════════════════════════════════════════════════════════"

