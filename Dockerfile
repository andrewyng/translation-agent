FROM python:3.9-bullseye

RUN mkdir /app
WORKDIR /app
COPY ./src /app/src
COPY ./pyproject.toml /app/pyproject.toml
COPY ./README.md /app/README.md

# install translation-agent with latest code
RUN pip install .

EXPOSE 8000

CMD ["python3", "-m", "translation_agent.web_api", "--port",  "8000"]