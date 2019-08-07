.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3
config:
	export AWS_ACCESS_KEY_ID=AKIAYMP3GYAWPVJ6MYQ5;\
	export AWS_SECRET_ACCESS_KEY=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

	
#################################################################################
# GLOBALS                                                                       #
#################################################################################
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = prj-sekiwa
PROFILE = rnddev 
PROJECT_NAME =sekiwa
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif
#################################################################################
# COMMANDS                                                                      #
#################################################################################
## Delete all compiled Python files and outfile
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf *.png

## Lint using flake8
lint:
	flake8 src

sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif
