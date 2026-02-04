FROM python:3.13-slim

# This sets the working directory inside the container
WORKDIR /app

# This will install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# This will copy the requirements.txt and install it
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# This will copy Popo's code
COPY . .

# This opens the port for Streamlit
EXPOSE 8501

# This tells Docker to check if Popo is healthy
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# This launches the app on container start
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
