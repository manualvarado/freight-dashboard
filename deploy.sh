#!/bin/bash

echo "🚀 Freight Dashboard Deployment Script"
echo "======================================"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial freight dashboard commit"
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Check if remote exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "🌐 Please add your GitHub repository as remote origin:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/freight-dashboard.git"
    echo "   git push -u origin main"
else
    echo "✅ Remote origin already configured"
fi

echo ""
echo "📋 Deployment Options:"
echo "1. Streamlit Cloud (Recommended - Free)"
echo "2. Heroku"
echo "3. Docker"
echo "4. Manual deployment"

read -p "Choose deployment option (1-4): " choice

case $choice in
    1)
        echo "🌐 Streamlit Cloud Deployment"
        echo "============================"
        echo "1. Push your code to GitHub:"
        echo "   git push origin main"
        echo ""
        echo "2. Go to: https://share.streamlit.io/"
        echo "3. Sign in with GitHub"
        echo "4. Click 'New app'"
        echo "5. Select your repository"
        echo "6. Set main file path: main.py"
        echo "7. Click 'Deploy'"
        echo ""
        echo "Your dashboard will be live at: https://your-app-name.streamlit.app"
        ;;
    2)
        echo "☁️ Heroku Deployment"
        echo "===================="
        echo "1. Install Heroku CLI:"
        echo "   brew install heroku/brew/heroku"
        echo ""
        echo "2. Login to Heroku:"
        echo "   heroku login"
        echo ""
        echo "3. Create and deploy:"
        echo "   heroku create your-freight-dashboard"
        echo "   git push heroku main"
        echo "   heroku open"
        ;;
    3)
        echo "🐳 Docker Deployment"
        echo "==================="
        echo "1. Build Docker image:"
        echo "   docker build -t freight-dashboard ."
        echo ""
        echo "2. Run container:"
        echo "   docker run -p 8501:8501 freight-dashboard"
        echo ""
        echo "3. Access at: http://localhost:8501"
        ;;
    4)
        echo "📖 Manual Deployment"
        echo "==================="
        echo "See DEPLOYMENT.md for detailed instructions"
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "✅ Deployment instructions completed!"
echo "📚 For detailed instructions, see DEPLOYMENT.md" 