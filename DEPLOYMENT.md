# 🚀 Deployment Guide

## Option 1: Streamlit Cloud (Recommended)

### Step 1: Prepare Your Code
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial freight dashboard commit"

# Create GitHub repository
# Go to github.com and create a new repository
# Then push your code:
git remote add origin https://github.com/YOUR_USERNAME/freight-dashboard.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `main.py`
6. Click "Deploy"

**Your dashboard will be live at**: `https://your-app-name.streamlit.app`

---

## Option 2: Heroku Deployment

### Step 1: Install Heroku CLI
```bash
# macOS
brew install heroku/brew/heroku

# Or download from: https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Deploy
```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-freight-dashboard

# Deploy
git push heroku main

# Open the app
heroku open
```

### Step 3: Configure (Optional)
```bash
# Set environment variables
heroku config:set STREAMLIT_SERVER_PORT=8501
heroku config:set STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Add custom domain (if you have one)
heroku domains:add yourdomain.com
```

---

## Option 3: Docker Deployment

### Step 1: Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build and Run
```bash
# Build image
docker build -t freight-dashboard .

# Run container
docker run -p 8501:8501 freight-dashboard
```

---

## Option 4: AWS/GCP/Azure Cloud

### AWS Elastic Beanstalk
1. Create `Procfile` (already created)
2. Create `.ebextensions/streamlit.config`
3. Deploy via AWS Console or CLI

### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT/freight-dashboard
gcloud run deploy --image gcr.io/YOUR_PROJECT/freight-dashboard --platform managed
```

---

## 🔒 Security Considerations

### 1. Authentication
```python
# Add to main.py for basic authentication
import streamlit_authenticator as stauth

# Create login
names = ['Admin', 'User1', 'User2']
usernames = ['admin', 'user1', 'user2']
passwords = ['password123', 'password456', 'password789']

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    'some_cookie_name', 'some_key', cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login', 'main')
```

### 2. Environment Variables
```bash
# Create .env file for sensitive data
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
DATABASE_URL=your_database_url
API_KEYS=your_api_keys
```

### 3. HTTPS/SSL
- Streamlit Cloud: Automatic HTTPS
- Heroku: Automatic HTTPS
- Custom domains: Configure SSL certificates

---

## 📊 Performance Optimization

### 1. Caching
```python
@st.cache_data
def load_data(uploaded_file):
    # Your data loading function
    pass
```

### 2. Database Connection
```python
# For PostgreSQL connection
import psycopg2
from sqlalchemy import create_engine

@st.cache_resource
def get_database_connection():
    return create_engine(os.getenv('DATABASE_URL'))
```

### 3. Large Data Handling
```python
# Chunk processing for large files
def process_large_csv(file, chunk_size=10000):
    for chunk in pd.read_csv(file, chunksize=chunk_size):
        # Process chunk
        yield chunk
```

---

## 🔧 Monitoring & Maintenance

### 1. Health Checks
```python
# Add to main.py
if st.button("Health Check"):
    st.success("Dashboard is running!")
```

### 2. Error Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Your code
    pass
except Exception as e:
    logger.error(f"Error: {e}")
    st.error("An error occurred. Please try again.")
```

### 3. Performance Monitoring
- Use Streamlit's built-in performance metrics
- Monitor memory usage
- Track user interactions

---

## 🌐 Custom Domain Setup

### 1. Streamlit Cloud
1. Go to app settings
2. Add custom domain
3. Update DNS records

### 2. Heroku
```bash
heroku domains:add yourdomain.com
# Follow DNS configuration instructions
```

---

## 📈 Scaling Considerations

### 1. Load Balancing
- Use multiple instances
- Configure auto-scaling
- Monitor resource usage

### 2. Database Scaling
- Use managed database services
- Implement connection pooling
- Optimize queries

### 3. Caching Strategy
- Redis for session data
- CDN for static assets
- Application-level caching

---

## 🆘 Troubleshooting

### Common Issues:
1. **Import Errors**: Check requirements.txt
2. **Port Issues**: Verify port configuration
3. **Memory Issues**: Optimize data processing
4. **Timeout Issues**: Increase timeout limits

### Support:
- Check Streamlit documentation
- Review deployment logs
- Test locally first 