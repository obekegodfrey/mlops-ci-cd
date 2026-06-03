# MLOps Project — Step 1: Raw Materials (Dataset & Code)

Welcome to the step of the MLOps pipeline. This repository sets up a lightweight, robust ML application using **FastAPI** to serve a **Logistic Regression** model trained on the classic **Iris dataset**. 

The design maintains a minimal footprint, keeping dependencies and the final Docker image optimized for standard cloud deployment environments (such as the AWS Free Tier limits).

---

## 1.1 Project Structure

```text
mlops-app/
├── src/
│   ├── model.py         # Train & save model
│   ├── predict.py       # Load model & predict
│   └── app.py           # FastAPI app
├── requirements.txt
└── tests/
    └── test_app.py      # Simple unit test
```

## 1.2 Application Source Code
### 1.2.1 src/model.py (Train & Save)

This script loads the Iris dataset, performs a standard train-test split, trains a Logistic Regression model, and serializes the trained model object using joblib. 

```
# src/model.py
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
<
BASE_DIR = Path(__file__).resolve().parent

def train_and_save_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

   
    model_path = BASE_DIR / "iris_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model trained and saved as {model_path}")

if __name__ == "__main__":
    train_and_save_model()

```
### 1.2.2 src/predict.py (Load & Predict)

This module handles loading the saved model file and wrapping the structural transformations necessary to parse inference payloads into single numerical class outputs.

```
# src/predict.py
import joblib
import numpy as np

def predict(input_data):
    model = joblib.load("iris_model.pkl")
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    return int(prediction[0])
```
### 1.2.3 src/app.py (FastAPI Application Layers)

An asynchronous API framework designed to expose health checks and operational prediction endpoints using explicit pydantic structural schemas.

```
# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "ML Model API is running "}

@app.post("/predict")
def get_prediction(data: InputData):
    result = predict(data.features)
    return {"prediction": result}

```

## 1.2.4 Requirements

The deterministic package locks used to build this environment configuration:

```
fastapi==0.115.0
uvicorn==0.30.0
scikit-learn==1.5.0
joblib==1.4.0
pydantic==2.9.0
pytest==8.0.0
```
## 1.2.5 Run Locally and Test

Follow these operational steps to build your local workspace, initialize training, and serve the application container.

### 1.2.5.1 Environment Provisioning

Initialize your virtual environment and resolve dependencies:

```
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment 0n linux
source .venv/bin/activate

# Upgrade package manager and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
### 1.2.5.2 Execute Training Pipeline

Execute the operational training loop to generate your localized binary model asset (iris_model.pkl):
```
python src/model.py
```
### 1.2.5.3 Boot Up the Web Server

Launch the FastAPI microservice engine locally. Ensuring the root working path variable is recognized correctly, start Uvicorn using:
```
uvicorn src.app:app --reload --port 8000
```
### 1.2.5.4 Interface Testing and API Verification

. Root Endpoint Verification: Open   your browser or execute a curl call against http://127.0.0.1:8000/.

. Interactive Documentation Interface: Navigate to http://127.0.0.1:8000/docs to interactively inspect payloads, execute standard /predict requests, and evaluate JSON model responses.

### Example Payloads:

{
“features”: [6.5, 3.0, 5.2, 2.0]
}

{
“features”: [5.1, 3.5, 1.4, 0.2]
}

## Step 2: Dockerize the App (Packaging Unit)

Just like a factory standardizes products with packaging, we need to package our ML app into a Docker container. This ensures it runs the same way everywhere — locally, in testing, or in production on AWS EC2.

### 2.1 Create Dockerfile in local machine
```
FROM python:3.12-slim

#Set the working directory
WORKDIR /app

#Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy the source code
COPY src/ ./src/   
COPY iris_model.pkl /app/iris_model.pkl

#Expose the port for the FastAPI app
EXPOSE 8000 

#Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.1.2 Create Dockerfile Distroless Base Image in local machine
```
# ==========================================
# STAGE 1: Build & Dependency Installation
# ==========================================
FROM python:3.12-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build-essential or extra system dependencies if packages require compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies into a dedicated site-packages directory
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ==========================================
# STAGE 2: Final Secure Distroless Runtime
# ==========================================
# Google's Python 3 distroless image is based on Debian
FROM gcr.io/distroless/python3-debian12

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source code and the pre-trained model artifact
COPY src/ ./src/
COPY iris_model.pkl /app/iris_model.pkl

# Expose the application port
EXPOSE 8000

# Set Python path to ensure the container can resolve app imports smoothly
ENV PYTHONPATH=/app

# Command to execute the application
# Note: Because distroless lacks a shell (like sh or bash), we use the vector/exec form 
# and point directly to the uvicorn module inside the python interpreter.
CMD ["-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

```
### 2.1.3 command to run docker and Test locally
```
#builds a Docker image from the Dockerfile in the current directory (`.`) and tags it with the name `mlops_ci_cd_iris`. This makes it easy to reference and run the image later.
docker build -t mlops_ci_cd .
```
```
#runs a container from the image, assigns it the name `mlops_ci_cd_iris`, and maps port 8000 inside the container to port 8000 on the host so the app is accessible via http://localhost:8000.
docker run --name mlops_ci_cd -p 8000:8000 mlops_ci_cd_iris
```
## Step 3: Quality Control → CI with GitHub Actions

In a factory, every product goes through quality checks before shipping. In our MLOps pipeline, that’s handled with Continuous Integration (CI).

### 3.1 GitHub Actions to automatically:

    . Set up Python
    . Install dependencies
    . Run tests (if present)
    . Build the Docker image

```
name: CI Pipeline   # Name of the workflow

on:
    push:
        branches: [ "main" ]   # Trigger the workflow on push to the main branch
    pull_request:
        branches: [ "main" ]   # Trigger the workflow on pull request to the main branch

permissions:
    contents: write   # Grant write permissions to the contents of the repository
    packages: write   # Grant write permissions to the packages in the repository

jobs:
    build:
        runs-on: ubuntu-latest   # Use the latest version of Ubuntu as the runner

        steps:
            # Step 1: Check out the code, set up Python, install dependencies, and run tests
            - name: Checkout code
              uses: actions/checkout@v3   # Check out the repository code
            # Step 2: Set up Python, install dependencies, and run tests
            - name: Set up Python
              uses: actions/setup-python@v4   # Set up Python environment
              with:
                  python-version: '3.12'   # Specify the Python version to use
            # Step 3: Install dependencies and run tests
            - name: Install dependencies
              run: |
                  python -m pip install --upgrade pip   # Upgrade pip
                  pip install -r requirements.txt   # Install dependencies from requirements.txt
                  pip install pytest   # Install pytest for testing
            # Step 4: Run tests
            - name: Run tests
              run: |
                  pytest tests/   # Run tests using pytest
                  pytest --maxfail=1 --disable-warnings -q || echo "Tests failed or no tests found, skipping..."   # Run tests with specific options
            # Step 5: Build  Docker image
            - name: Build Docker image
              run: |
                  docker build -t mlops-ci-cd .   # Build a Docker image
            # Step 6: Log in to Docker Hub
            - name: Log in to Docker Hub
              uses: docker/login-action@v2   # Log in to Docker Hub
              with:
                  username: ${{ secrets.DOCKER_USERNAME }}   # Use Docker Hub username from secrets
                  password: ${{ secrets.DOCKER_PASSWORD }}   # Use Docker Hub password from secrets
            # Step 7: Push Docker image to Docker Hub
            - name: Push Docker image
              run: |
                  docker tag mlops-ci-cd:latest ${{ secrets.DOCKER_USERNAME }}/mlops-ci-cd:latest   # Tag the Docker image
                  docker push ${{ secrets.DOCKER_USERNAME }}/mlops-ci-cd:latest   # Push the Docker image to Docker Hub     
                                
```
### 3.2 Commands to Trigger CI
```
# Stage all changes in your repo
git add .

# Commit changes with a descriptive message
git commit -m "Added test step in CI/CD pipeline"

# Push changes to the main branch (triggers CI/CD pipeline)
git push origin main
```
