# Use rocker base image with R preinstalled
FROM rocker/r-ver:4.3.1

# Copy specific version of uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock* /app/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    libssl-dev \
    libffi-dev \
    libcurl4-openssl-dev \
    libpcre2-dev \
    liblzma-dev \
    libbz2-dev \
    zlib1g-dev \
    libicu-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install R package 'sirus'
RUN R -e "install.packages('sirus', repos='https://cloud.r-project.org')"

# Sync Python dependencies using uv
RUN uv sync --locked

# Copy the rest of your application
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Default command: run the app using uv
CMD ["uv", "run", "streamlit", "run", "app.py"]
