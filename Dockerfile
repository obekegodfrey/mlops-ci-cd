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