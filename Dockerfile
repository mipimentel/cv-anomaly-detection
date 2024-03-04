FROM python:3.10

ENV CODE_DIR /code
ENV PYTHONPATH "${PYTHONPATH}:${CODE_DIR}"

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 graphviz  -y

RUN useradd --create-home --home-dir $CODE_DIR user \
    && chmod -R u+rwx $CODE_DIR \
    && chown -R user:user $CODE_DIR

ENV PYTHONUNBUFFERED=1
WORKDIR $CODE_DIR
COPY pyproject.toml /code/
COPY poetry.lock /code/

RUN pip3 install poetry
RUN poetry config virtualenvs.create false

RUN poetry install --no-root

USER user

EXPOSE 8888
