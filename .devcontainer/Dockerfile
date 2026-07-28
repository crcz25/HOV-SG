# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
ARG DEBIAN_FRONTEND=noninteractive

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    CONDA_DEFAULT_ENV=hovsg \
    PATH=/opt/conda/envs/hovsg/bin:/opt/conda/bin:${PATH} \
    HOME=/home/${USERNAME} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    bzip2 \
    ca-certificates \
    curl \
    git \
    graphviz \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgraphviz-dev \
    libice6 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    mesa-utils \
    nano \
    pkg-config \
    sudo \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN if ! getent group "${USERNAME}" >/dev/null; then \
      if getent group "${USER_GID}" >/dev/null; then \
        groupadd "${USERNAME}"; \
      else \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
      fi; \
    fi \
    && if id -u "${USERNAME}" >/dev/null 2>&1; then \
      usermod -u "${USER_UID}" "${USERNAME}" || true; \
      usermod -g "${USERNAME}" "${USERNAME}"; \
      usermod -d "/home/${USERNAME}" -m "${USERNAME}" || true; \
      usermod -s /bin/bash "${USERNAME}"; \
    elif getent passwd "${USER_UID}" >/dev/null; then \
      EXISTING_USER="$(getent passwd "${USER_UID}" | cut -d: -f1)"; \
      usermod -l "${USERNAME}" "${EXISTING_USER}"; \
      usermod -g "${USERNAME}" "${USERNAME}"; \
      usermod -d "/home/${USERNAME}" -m "${USERNAME}" || true; \
      usermod -s /bin/bash "${USERNAME}"; \
    else \
      useradd --uid "${USER_UID}" --gid "${USERNAME}" -m "${USERNAME}" -s /bin/bash; \
    fi \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" >/etc/sudoers.d/"${USERNAME}" \
    && chmod 0440 /etc/sudoers.d/"${USERNAME}"

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && mkdir -p "${MAMBA_ROOT_PREFIX}" \
    && chown -R "${USERNAME}:${USERNAME}" "${MAMBA_ROOT_PREFIX}"

COPY environment.yaml /tmp/environment.yaml

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba config set remote_connect_timeout_secs 60 \
    && micromamba config set remote_read_timeout_secs 600 \
    && micromamba config set remote_max_retries 5 \
    && micromamba env create -y -p /opt/conda/envs/hovsg -f /tmp/environment.yaml \
    && micromamba run -p /opt/conda/envs/hovsg python -m pip check \
    && micromamba clean -a -y \
    && chown -R "${USERNAME}:${USERNAME}" "${MAMBA_ROOT_PREFIX}"

RUN python -m pip install --no-cache-dir wn

USER "${USERNAME}"

RUN curl -L -o /tmp/oewn-2025-plus.xml.gz \
    https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025-plus.xml.gz \
    && (python -c "import wn; wn.add('/tmp/oewn-2025-plus.xml.gz')" \
        || python -c "import wn; lexicon = 'oewn:2025+'; assert any(f'{item.id}:{item.version}' == lexicon for item in wn.lexicons()), 'OEWN add failed and the lexicon is not registered'; print(f'{lexicon} is already registered; continuing')") \
    && rm -f /tmp/oewn-2025-plus.xml.gz

USER root

COPY .devcontainer/entrypoint.sh /usr/local/bin/devcontainer-entrypoint
RUN chmod +x /usr/local/bin/devcontainer-entrypoint \
    && mkdir -p /workspace/HOV-SG \
    && echo 'export PATH="/opt/conda/envs/hovsg/bin:/opt/conda/bin:${PATH}"' >>"/home/${USERNAME}/.bashrc" \
    && echo 'export CONDA_DEFAULT_ENV="hovsg"' >>"/home/${USERNAME}/.bashrc" \
    && echo 'export PYTHONPATH="/workspace/HOV-SG:${PYTHONPATH:-}"' >>"/home/${USERNAME}/.bashrc" \
    && chown -R "${USERNAME}:${USERNAME}" /workspace "/home/${USERNAME}/.bashrc"

USER "${USERNAME}"
WORKDIR /workspace/HOV-SG

ENTRYPOINT ["devcontainer-entrypoint"]
CMD ["sleep", "infinity"]
