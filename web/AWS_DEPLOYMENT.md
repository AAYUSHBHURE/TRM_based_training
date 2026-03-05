# TRM Web App - AWS S3 Deployment Guide

## 🚀 Deployment Steps

### 1. Build Static Export

Since the local build has issues, use **GitHub Actions** to build in the cloud:

#### Option A: GitHub Actions (Recommended)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to S3

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'
          
      - name: Install dependencies
        working-directory: ./web
        run: npm ci
        
      - name: Build Next.js
        working-directory: ./web
        run: npm run build
        
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
          
      - name: Deploy to S3
        working-directory: ./web
        run: aws s3 sync out/ s3://your-bucket-name --delete
```

#### Option B: Build on Linux Machine/WSL

```bash
# If you have WSL or can use a Linux VM
cd web
npm install
npm run build
# The 'out/' folder contains your static site
```

#### Option C: Deploy HTML Demo Instead

The `demo.html` is already built and ready:
```bash
# Just upload demo.html to S3
aws s3 cp demo.html s3://your-bucket-name/index.html
```

---

### 2. Create S3 Bucket

```bash
# Create bucket
aws s3 mb s3://trm-demo --region us-east-1

# Enable static website hosting
aws s3 website s3://trm-demo \
  --index-document index.html \
  --error-document index.html
```

---

### 3. Set Bucket Policy (Public Access)

Create `bucket-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::trm-demo/*"
    }
  ]
}
```

Apply policy:
```bash
aws s3api put-bucket-policy \
  --bucket trm-demo \
  --policy file://bucket-policy.json
```

---

### 4. Upload Files

#### If using Next.js build:
```bash
cd web
aws s3 sync out/ s3://trm-demo --delete
```

#### If using HTML demo:
```bash
aws s3 cp demo.html s3://trm-demo/index.html \
  --content-type "text/html"
```

---

### 5. Access Your Site

Website URL:
```
http://trm-demo.s3-website-us-east-1.amazonaws.com
```

---

## 🌐 Optional: CloudFront CDN

For HTTPS and better performance:

```bash
# Create CloudFront distribution
aws cloudfront create-distribution \
  --origin-domain-name trm-demo.s3.amazonaws.com \
  --default-root-object index.html
```

---

## 📋 Quick Commands Reference

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Sync files
aws s3 sync ./web/out s3://trm-demo --delete

# View website
echo "http://trm-demo.s3-website-us-east-1.amazonaws.com"
```

---

## ⚡ Fastest Option for Presentation

**Use the HTML demo:**
1. Already built ✅
2. No build issues ✅
3. Works perfectly ✅

```bash
aws s3 cp demo.html s3://trm-demo/index.html
```

Done in 30 seconds! 🚀
