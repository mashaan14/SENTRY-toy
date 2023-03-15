"""Params for SENTRY"""

# params for the dataset
num_classes = 2
num_neighbors = 7
queue_length = 128

# params for training network
num_gpu = 1
num_epochs_pre = 1 #10
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
num_epochs = 10 #100
log_step = 100
save_step = 100
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9

# Loss weights
lambda_CE = 1.0 # 1.0
lambda_IE = 0.1 #0.1
lambda_SENTRY = 1.0 #1.0
