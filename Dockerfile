# Reproducible environment for the sgmacro workflow.
#
# The image carries the committed snapshot store and LLM response cache, so the
# default command regenerates the sample report with no network and no keys.
# Add --refetch (and drop --offline) to pull fresh data instead.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency metadata first, so a source edit does not invalidate the pip layer.
COPY pyproject.toml README.md constraints.txt ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]" -c constraints.txt

COPY config ./config
COPY data ./data
COPY runs ./runs
COPY reports ./reports
COPY llm_cache ./llm_cache
COPY discovery ./discovery
COPY docs ./docs
COPY tests ./tests

# Fail the build if the workflow cannot reproduce its own committed report offline.
RUN pytest -q

ENTRYPOINT ["sgmacro"]
CMD ["run", "--as-of", "2026-10-02", "--offline", "--llm", "cache"]
