FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code and the model directory
COPY ./app /app

# Set the working directory
WORKDIR /app

# Expose the port that the app runs on
EXPOSE 8006

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8006"]
