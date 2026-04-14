# 1. Use an official Python environment
FROM python:3.11-slim

# 2. Install the system libraries that were crashing on Streamlit
# These are the "C++" parts that OpenCV and MediaPipe need to work.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the "home folder" inside the server
WORKDIR /app

# 4. Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your app's code
COPY . .

# 6. Tell the server to use port 8501 (Streamlit's default)
EXPOSE 8501

# 7. The command to start your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
