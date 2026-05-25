FROM texlive/texlive:latest

ARG UID=1000
ARG GID=1000
ARG USERNAME=dev

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    biber \
    ghostscript \
    inkscape \
    less \
    perl \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN tlmgr install keytheorems

RUN set -eux; \
    if getent group "${GID}" >/dev/null; then \
        GROUP_NAME="$(getent group "${GID}" | cut -d: -f1)"; \
    else \
        groupadd --gid "${GID}" "${USERNAME}"; \
        GROUP_NAME="${USERNAME}"; \
    fi; \
    if id -u "${USERNAME}" >/dev/null 2>&1; then \
        usermod --uid "${UID}" --gid "${GID}" "${USERNAME}"; \
    elif getent passwd "${UID}" >/dev/null; then \
        EXISTING_USER="$(getent passwd "${UID}" | cut -d: -f1)"; \
        usermod --login "${USERNAME}" --home "/home/${USERNAME}" --move-home --gid "${GID}" "${EXISTING_USER}"; \
    else \
        useradd --uid "${UID}" --gid "${GROUP_NAME}" --create-home --shell /bin/bash "${USERNAME}"; \
    fi

USER ${USERNAME}
WORKDIR /workspace/report

ENTRYPOINT ["tini", "-s", "--"]
CMD ["bash"]
