# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install git CLI
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN useradd -ms /bin/bash vscode
USER vscode

# Copy the requirements file into the container at /app
COPY --chown=vscode:vscode requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Add the user's .local/bin to the PATH
ENV PATH="/home/vscode/.local/bin:${PATH}"