import subprocess

#target_gini should range between 0 and 1
target_gini = 0.5
dataset = 'MNIST'
epoch = 10

cmd1 = ['python', 'C:/Users/jesse/OneDrive/Documenten/GitHub/MScThesis-Sparsity-and-Quickselection/QuickSelection-main/QuickSelection/train_sparse_DAE.py', '--dataset_name', dataset, '--epoch', str(epoch)]
cmd2 = ['python', 'C:/Users/jesse/OneDrive/Documenten/GitHub/MScThesis-Sparsity-and-Quickselection/QuickSelection-main/QuickSelection/QuickSelection.py', '--dataset_name', dataset, '--epoch', str(epoch)]

subprocess.run(cmd1)
subprocess.run(cmd2)