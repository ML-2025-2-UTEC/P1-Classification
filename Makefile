# Makefile para el Proyecto de Machine Learning

# Variables
PYTHON = python3
PIP = pip3
SRC_DIR = src
TEST_DIR = tests
REPORTS_DIR = reports
DATA_RAW_DIR = data/raw
DATA_PROCESSED_DIR = data/processed
MODELS_DIR = models
EXPERIMENTS_DIR = experiments

# Evitar que 'clean' se confunda con un archivo llamado 'clean'
.PHONY: all install test clean eda preprocess train evaluate

# Comando por defecto
all: install test

# Instalar dependencias
install:
	$(PIP) install -r requirements.txt

# Ejecutar tests
test:
	$(PYTHON) -m pytest $(TEST_DIR)

# Limpiar artefactos generados
clean:
	rm -rf $(DATA_PROCESSED_DIR)/*
	rm -rf $(MODELS_DIR)/*
	rm -rf $(REPORTS_DIR)/*
	rm -rf $(EXPERIMENTS_DIR)/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Ejecutar pipeline del proyecto
eda:
	$(PYTHON) -m $(SRC_DIR).eda.eda_report --input datos_entrenamiento_riesgo.csv

preprocess:
	$(PYTHON) -m $(SRC_DIR).features.build_features

train:
	$(PYTHON) -m $(SRC_DIR).models.train_manual --model logistic

evaluate:
	$(PYTHON) -m $(SRC_DIR).evaluation.evaluate_model --model_path $(MODELS_DIR)/logistic_manual.joblib

run_pipeline: eda preprocess train evaluate

help:
	@echo "Comandos disponibles:"
	@echo "  install      - Instala las dependencias del proyecto"
	@echo "  test         - Ejecuta las pruebas unitarias"
	@echo "  clean        - Elimina todos los archivos generados"
	@echo "  eda          - Ejecuta el script de Análisis Exploratorio de Datos"
	@echo "  preprocess   - Ejecuta el script de preprocesamiento de datos"
	@echo "  train        - Entrena un modelo (ej. make train model=rf)"
	@echo "  evaluate     - Evalúa un modelo entrenado"
	@echo "  run_pipeline - Ejecuta el pipeline completo: eda, preprocess, train, evaluate"
	@echo "  help         - Muestra esta ayuda"