FROM dolfinx/dolfinx:v0.9.0 

WORKDIR /app

USER root

RUN apt-get update && apt-get install -y \
    libgl1 \
    libxrender1 \
    libxext6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

