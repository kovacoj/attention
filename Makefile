COMPOSE := docker compose -f containers/compose.yaml

.PHONY: containers-build latex-shell report

containers-build:
	$(COMPOSE) build

latex-shell:
	$(COMPOSE) run --rm latex bash

report:
	./bin/latex-build
