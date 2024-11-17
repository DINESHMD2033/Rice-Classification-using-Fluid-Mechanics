# Base image
FROM python:3.7

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt /app
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the entire project to the container
COPY . /app

# Expose the application port
EXPOSE 8080

# Command to start the application
CMD ["python", "app.py"]
