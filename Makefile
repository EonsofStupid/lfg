SHELL := /bin/sh

IMAGE_NAME ?= lfg:latest
CONTAINER_NAME ?= lfg-server
HOST_PORT ?= 4317
BIND_ADDR ?= 127.0.0.1:4317
PYTHON ?= python3
BUILD_VERSION ?= 0.0.0-dev
BUILD_REVISION ?= $(shell git rev-parse --verify HEAD 2>/dev/null || echo unknown)
BUILD_CREATED ?= $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
SBOM_PATH ?= output/sbom/lfg.spdx

.PHONY: help build-binary build-ui run-binary lfg-bench ml-install ml-validate ml-distill-template ml-distill-validate ml-distill-build image-build image-run image-stop image-sbom sbom-baseline-dry-run

help:
	@echo "Targets:"
	@echo "  build-binary          Build Rust release binary (lfg-server)"
	@echo "  build-ui              Build React UI bundle"
	@echo "  run-binary            Run lfg-server serve locally"
	@echo "  lfg-bench             Run encoder benchmark"
	@echo "  ml-install            Install Python deps for ml/lfg-encoder"
	@echo "  ml-validate           Validate synthetic classifier dataset"
	@echo "  ml-distill-template   Create teacher trace template"
	@echo "  ml-distill-validate   Validate teacher traces JSONL"
	@echo "  ml-distill-build      Build SFT train/eval JSONL splits"
	@echo "  image-build           Build deployment image (includes python3 + deps)"
	@echo "  image-sbom            Generate image SBOM (SPDX JSON)"
	@echo "  sbom-baseline-dry-run Show proposed SBOM baseline update from current SBOM"
	@echo "  image-run             Run containerized lfg-server"
	@echo "  image-stop            Stop and remove running container"

build-binary:
	cargo build --release -p lfg-server

build-ui:
	cd ui/master-clyffy-web && npm ci && npm run build

run-binary:
	cargo run -p lfg-server -- serve --bind $(BIND_ADDR)

lfg-bench:
	cargo run -p lfg-server -- lfg-bench --iterations 1000

ml-install:
	$(PYTHON) -m pip install -r ml/lfg-encoder/requirements.txt

ml-validate:
	$(PYTHON) ml/lfg-encoder/train.py validate

ml-distill-template:
	$(PYTHON) ml/lfg-encoder/distill.py init-template

ml-distill-validate:
	$(PYTHON) ml/lfg-encoder/distill.py validate-traces --traces ml/lfg-encoder/dataset/teacher-traces.v1.jsonl

ml-distill-build:
	$(PYTHON) ml/lfg-encoder/distill.py build-sft --traces ml/lfg-encoder/dataset/teacher-traces.v1.jsonl

image-build:
	docker build \
		--build-arg BUILD_VERSION=$(BUILD_VERSION) \
		--build-arg BUILD_REVISION=$(BUILD_REVISION) \
		--build-arg BUILD_CREATED=$(BUILD_CREATED) \
		-t $(IMAGE_NAME) .

image-sbom:
	mkdir -p $(dir $(SBOM_PATH))
	if command -v syft >/dev/null 2>&1; then \
		syft $(IMAGE_NAME) -o spdx=$(SBOM_PATH); \
	else \
		docker scout sbom --format spdx --output $(SBOM_PATH) $(IMAGE_NAME); \
	fi

sbom-baseline-dry-run:
	python scripts/refresh-sbom-policy-baseline.py --sbom output/sbom/lfg-policy.json --policy security/sbom-policy.json

image-run:
	docker run --rm -d --name $(CONTAINER_NAME) -p $(HOST_PORT):4317 $(IMAGE_NAME)

image-stop:
	-docker stop $(CONTAINER_NAME)
