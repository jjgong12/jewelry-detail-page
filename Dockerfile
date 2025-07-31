FROM python:3.10-slim

WORKDIR /

# Install Python dependencies only
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler
COPY handler.py /handler.py

# Start the handler
CMD ["python", "-u", "/handler.py"]
