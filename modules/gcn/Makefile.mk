help::
	@echo "GCN using PyTorch (gcn):"
	@echo "  lsp-cond-gen-graph-data creates the data for the experiments."
	@echo "  lsp-cond-train-autoencoder trains the auto encoder model."
	@echo ""


BASENAME ?= gcn
NUM_TRAINING_SEEDS ?= 2000
NUM_TESTING_SEEDS ?= 500
NUM_EVAL_SEEDS ?= 500
DATA_SET = cora

CORE_ARGS ?= --save_dir /data/ \
		--seed 42 \
		--dataset $(DATA_SET)

DATA_GEN_ARGS ?= $(CORE_ARGS) \
		--save_dir /data/$(BASENAME)/ 

TRAINING_ARGS ?= --test_log_frequency 10 \
		--save_dir /data/$(BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(BASENAME)/ \
		--lsp_weight $(LSP_WEIGHT) \
		--loc_weight $(LOC_WEIGHT) \
		--loss $(LOSS) \
		--relative_positive_weight $(RPW) \
		--input_type $(INPUT_TYPE)

EVAL_ARGS ?= $(CORE_ARGS) \
		--save_dir /data/$(BASENAME)/results/$(EXPERIMENT_NAME) \
		--network_file /data/$(BASENAME)/logs/$(EXPERIMENT_NAME)/model.pt


run1-file = $(DATA_BASE_DIR)/$(BASENAME)/logs/$(EXPERIMENT_NAME)/run1.pt 
$(run1-file):
	@$(DOCKER_PYTHON) -m gcn.scripts.run \
		$(CORE_ARGS) \
		--run1

.PHONY: run-1
run-1: DOCKER_ARGS ?= -it
run-1: $(run1-file)


run2-file = $(DATA_BASE_DIR)/$(BASENAME)/logs/$(EXPERIMENT_NAME)/run2.pt 
$(run2-file):
	@$(DOCKER_PYTHON) -m gcn.scripts.run \
		$(CORE_ARGS) \
		--run2

.PHONY: run-2
run-2: DOCKER_ARGS ?= -it
run-2: $(run2-file)


run3-file = $(DATA_BASE_DIR)/$(BASENAME)/logs/$(EXPERIMENT_NAME)/run3.pt 
$(run3-file):
	@$(DOCKER_PYTHON) -m gcn.scripts.run \
		$(CORE_ARGS) \
		--run3

.PHONY: run-3
run-3: DOCKER_ARGS ?= -it
run-3: $(run3-file)


multi-file = $(DATA_BASE_DIR)/$(BASENAME)/logs/$(EXPERIMENT_NAME)/multi.pt 
$(multi-file):
	@$(DOCKER_PYTHON) -m gcn.scripts.run \
		$(CORE_ARGS)

.PHONY: run-multi
run-multi: DOCKER_ARGS ?= -it
run-multi: $(multi-file)



train-file = $(DATA_BASE_DIR)/$(BASENAME)/logs/$(EXPERIMENT_NAME)/model.pt 
$(train-file):
	@$(DOCKER_PYTHON) -m gcn.scripts.train \
		$(CORE_ARGS) \
		--save_dir /data/$(BASENAME)/logs/$(EXPERIMENT_NAME) \
		--num_steps 12000 \
		--lr 5e-1 \
		--weight_decay .5 \
		--epochs 1000 \

.PHONY: train
train: DOCKER_ARGS ?= -it
train: $(train-file)


eval-seeds = /data/$(BASENAME)/results/$(EXPERIMENT_NAME)/logfile.txt
$(eval-seeds): $(train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(BASENAME)/results/$(EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m gcn.scripts.eval \
		$(EVAL_ARGS) \
	 	--logfile logfile.txt \
	 	--seed 420 \

.PHONY: eval
eval: $(eval-seeds)
# 	$(MAKE) lsp-cond-results


