CATEGORY ?= bottle
BACKEND ?= padim_resnet50
CONFIG ?= configs/default.yaml
EPOCHS ?= 30

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

data:
	python scripts/download_mvtec.py --config $(CONFIG)

splits:
	python scripts/prepare_splits.py --category $(CATEGORY) --config $(CONFIG)

train_padim:
	python -m src.train.fit_padim --category $(CATEGORY) --config $(CONFIG)

train_cae:
	python -m src.train.train_cae --category $(CATEGORY) --epochs $(EPOCHS) --config $(CONFIG)

eval:
	python -m src.eval.evaluate --backend $(BACKEND) --category $(CATEGORY) --config $(CONFIG)

app:
	python -m src.app.ui_gradio --config $(CONFIG)
