# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the moon directory into the container
COPY moon /app/moon

# Set default command
CMD ["python", "-m", "moon.main"] 