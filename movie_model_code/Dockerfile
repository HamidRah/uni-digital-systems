# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

COPY ./wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Install curl
RUN apt-get update && apt-get install -y curl

# Install any needed packages specified in requirements.txt with increased timeout and retries
RUN pip install --default-timeout=100 --retries=5 -r requirements.txt


# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV development

# Define environment variable
ENV NAME World

RUN chmod +x sql_scripts/

