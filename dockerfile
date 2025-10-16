# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .
RUN chmod -R 777 /app
USER root
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 5000

# Run the Streamlit application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
