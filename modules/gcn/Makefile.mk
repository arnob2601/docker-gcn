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
		--num_steps 10000 \
		--lr 1e-1 \
		--weight_decay .4 \
		--epochs 2000 \

.PHONY: train
train: DOCKER_ARGS ?= -it
train: $(train-file)

# lsp-cond-train-file = $(DATA_BASE_DIR)/$(BASENAME)/logs/$(EXPERIMENT_NAME)/model.pt 
# $(lsp-cond-train-file): $(lsp-cond-autoencoder-train-file)
# 	@$(DOCKER_PYTHON) -m lsp_cond.scripts.train \
# 		$(TRAINING_ARGS) \
# 		--num_steps 50000 \
# 		--learning_rate 1e-4 \
# 		--learning_rate_decay_factor .6 \
# 		--epoch_size 10000 \
# 		--autoencoder_network_file /data/$(BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt 

# .PHONY: lsp-cond-train lsp-cond-train-autoencoder lsp-cond-train-cnn lsp-cond-train-marginal
# lsp-cond-train-autoencoder: DOCKER_ARGS ?= -it
# lsp-cond-train-autoencoder: $(lsp-cond-autoencoder-train-file)
# lsp-cond-train-cnn: DOCKER_ARGS ?= -it
# lsp-cond-train-cnn: $(lsp-cond-cnn-train-file)
# lsp-cond-train-marginal: DOCKER_ARGS ?= -it
# lsp-cond-train-marginal: $(lsp-cond-marginal-train-file)
# lsp-cond-train: DOCKER_ARGS ?= -it
# lsp-cond-train: $(lsp-cond-train-file)


# PLOTTING_ARGS = \
# 		--data_file /data/$(BASENAME)/results/$(EXPERIMENT_NAME)/logfile.txt \
# 		--output_image_file /data/$(BASENAME)/results/results_$(EXPERIMENT_NAME).png

# .PHONY: lsp-cond-results
# lsp-cond-results:
# 	@$(DOCKER_PYTHON) -m lsp_cond.scripts.cond_lsp_plotting \
# 		$(PLOTTING_ARGS)
# .PHONY: lsp-cond-results-marginal
# lsp-cond-results-marginal:
# 	@$(DOCKER_PYTHON) -m lsp_cond.scripts.cond_lsp_plotting \
# 		--data_file /data/$(BASENAME)/results/$(EXPERIMENT_NAME)/mlsp_logfile.txt \
# 		--gnn

# lsp-cond-eval-seeds = \
# 	$(shell for ii in $$(seq 50000 $$((50000 + $(NUM_EVAL_SEEDS) - 1))); \
# 		do echo "$(DATA_BASE_DIR)/$(BASENAME)/results/$(EXPERIMENT_NAME)/maze_learned_$${ii}.png"; done)
# $(lsp-cond-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
# $(lsp-cond-eval-seeds): $(lsp-cond-cnn-train-file)
eval-seeds = /data/$(BASENAME)/results/$(EXPERIMENT_NAME)/logfile.txt
$(eval-seeds): $(train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(BASENAME)/results/$(EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m gcn.scripts.eval \
		$(EVAL_ARGS) \
	 	--logfile logfile.txt

.PHONY: eval
eval: $(eval-seeds)
# 	$(MAKE) lsp-cond-results


