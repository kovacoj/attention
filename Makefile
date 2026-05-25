COMPOSE := docker compose -f docker-compose.yaml

.PHONY: containers-build latex-shell report

containers-build:
	$(COMPOSE) build

latex-shell:
	$(COMPOSE) run --rm latex bash

report:
	@set -e; \
	if ! $(COMPOSE) run --rm latex latexmk paper.tex; then \
		$(COMPOSE) run --rm latex latexmk -g paper.tex; \
	fi
