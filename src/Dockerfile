# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable to specify where the Flask app is located
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run app.py using the Flask CLI
# CMD ["flask", "run"]

# Run as a module
CMD ["python", "-m", "flask", "run"] 


# # Modules Check
# Tell us quickly if we have any missing imports
# FROM python:3.9-slim

# WORKDIR /app

# # Copy only the script needed to check imports
# COPY check_imports.py /app/

# CMD ["python", "check_imports.py"]
