"""
Test interface for ECG classification!
You can just run this file.
"""
import os
import argparse
import torch
import options
import datetime
import random
import utils
import numpy as np
from dataloader import create_dataloader_for_test
import sklearn
######################################################################################################################
#                                                  Parser init                                                       #
######################################################################################################################
# Parse command-line arguments and configurations for the experiment
opt = options.Options().init(argparse.ArgumentParser(description='ECG classification')).parse_args()
print(opt)

######################################################################################################################
#                                    Set a model (check point) and a log folder                                      #
######################################################################################################################
# Get the absolute path of the current script and initialize directories for logging and models
dir_name = os.path.dirname(os.path.abspath(__file__))  # absolute path
print(dir_name)

log_dir = os.path.join(dir_name, 'log', opt.arch + '_' + opt.env)

utils.mkdir(log_dir)
print("Now time is : ", datetime.datetime.now().isoformat())
tboard_dir = './log/{}_{}/logs'.format(opt.arch, opt.env)  # os.path.join(log_dir, 'logs')
model_dir = './log/{}_{}/models'.format(opt.arch, opt.env)  # os.path.join(log_dir, 'models')
utils.mkdir(model_dir)  # make a dir if there is no dir (given path)
utils.mkdir(tboard_dir)

######################################################################################################################
#                                                   Model init                                                       #
######################################################################################################################
# Set the device (either CPU or GPU) and initialize the random seeds for reproducibility
DEVICE = torch.device(opt.device)

# set seeds
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# Initialize the model and print its size
model = utils.get_arch(opt)

total_params = utils.cal_total_params(model)
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))

# Load a pre-trained model
print('Load the pretrained model...')
chkpt = torch.load(opt.pretrain_model_path)
model.load_state_dict(chkpt['model'])

model = model.to(DEVICE)

######################################################################################################################
######################################################################################################################
#                                             Main program - test                                                    #
######################################################################################################################
######################################################################################################################
# Start testing
test_loader = create_dataloader_for_test(opt)
test_log_fp = open(model_dir + '/test_log.txt', 'a')

t_all = []
o_all = []
b_all = []

model.eval()
with torch.no_grad():
    for inputs, targets in utils.Bar(test_loader):
        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)
        q = torch.sigmoid(outputs)

        t_all.append(targets.data.cpu().numpy())
        o_all.append(q.data.cpu().numpy())

t_all = np.concatenate(t_all, axis=0)
o_all = np.concatenate(o_all, axis=0)

# Load the threshold to classify outputs (you can get the threshold using utils.find_threshold)
threshold = np.load('./threshold.npy')
print('Threshold: ', threshold)

# Convert the outputs to binary labels based on the threshold
for idx in range(len(o_all)):
    labels = utils.get_binary_outputs_np(o_all[idx], threshold)
    b_all.append(labels)

b_all = np.array(b_all)

######################################################################################################################
#                                                   Get scores                                                       #
######################################################################################################################
# Compute evaluation metrics (F-Measure, AUPRC, Hamming loss and Specificity) for the predictions
num_classes = opt.class_num

s_all = np.zeros((len(b_all), num_classes), dtype=np.float64)
for i in range(len(b_all)):
    s_all[i] = [True if v == 1 else False for v in b_all[i]]

auprc = utils.compute_auprc(t_all, s_all)
f_measure, specificity = utils.compute_f1_sp(t_all, b_all)
hamming_loss = sklearn.metrics.hamming_loss(t_all, b_all)
print("####################################################################")
print(opt.pretrain_model_path)
print("F-Measure : {:.4}".format(f_measure))
print("AUPRC : {:.4}".format(auprc))
print("Hamming loss : {:.4}".format(hamming_loss))
print("specificity (Sp) : {:.4}".format(specificity))
print("####################################################################")

print('System has been finished.')
