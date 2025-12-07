.PHONY: data train clean help

# Default target
help:
	@echo "Usage:"
	@echo "  make data     - Create raw parquet from backend_roles.json (run once)"
	@echo "  make train    - Run training (auto-formats prompts if needed)"
	@echo "  make clean    - Remove processed data (keeps raw)"

# Create raw parquet data from JSON (run once when adding new roles)
data: data/raw/train.parquet

data/raw/train.parquet: data/backend_roles.json
	python scripts/prepare_dataset.py

# Run training (auto-formats prompts if prompts.py changed)
train: data/raw/train.parquet
	python -m src.recruiter.main

# Clean processed data (keeps raw)
clean:
	rm -rf data/processed/

# Clean everything
clean-all:
	rm -rf data/raw/ data/processed/
