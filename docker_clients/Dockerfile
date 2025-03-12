FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    build-essential && \
    apt-get clean

# Copy requirements
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the client code
COPY . .

# Set default command (client name passed dynamically)
CMD ["sh", "-c", "CLIENT_NAME=${CLIENT_NAME} python client.py"]

# # Use an ARM-compatible base image
# FROM python:3.9-slim

# # Set the working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     python3-setuptools \
#     python3-wheel \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # Upgrade pip inside the container
# RUN pip install --upgrade pip

# # Copy requirements file and install dependencies
# COPY requirements.txt requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy project files
# COPY . .

# # Define entry point
# CMD ["python", "client.py"]
