# Use an ARM-compatible base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip inside the container
RUN pip install --upgrade pip

# Install Cython first to avoid issues
RUN pip install --no-cache-dir Cython==0.29.34 setuptools==59.5.0

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FL server script
COPY . .

# Expose port for communication with clients
EXPOSE 8080

# Run the server
CMD ["python", "server.py"]
