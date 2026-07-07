FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget curl unzip \
    bcftools tabix \
    python3 python3-pip python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PLINK2 (Linux x86_64 latest)
RUN wget -qO /tmp/plink2.zip https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_x86_64_latest.zip \
    && unzip /tmp/plink2.zip -d /usr/local/bin \
    && chmod +x /usr/local/bin/plink2 \
    && rm -f /tmp/plink2.zip

# Optional KING binary is not guaranteed in apt; pipeline falls back to plink2 king-table.

RUN pip3 install --no-cache-dir pandas numpy matplotlib

WORKDIR /work

CMD ["/bin/bash"]
