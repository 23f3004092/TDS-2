# Use the official, full Python 3.11 image (Debian-based)
# This is the "Big" version, not "slim" or "alpine".
FROM python:3.11

# Set the working directory
WORKDIR /code

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
# We upgrade pip first to avoid binary wheel issues with Pandas/Playwright
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

# Install Playwright browsers and system dependencies
# This step adds significant size but is required for the bot to "see" websites
RUN playwright install chromium && \
    playwright install-deps

# Create a non-root user (Hugging Face security requirement)
# We use user ID 1000 which is the standard for HF Spaces
RUN useradd -m -u 1000 user

# Copy the rest of the application code
COPY . /code

# Switch to the non-root user
USER user

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose the Hugging Face port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]