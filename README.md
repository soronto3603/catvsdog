# catvsdog
install anaconda
start anaconda-navigator
click environment
create new environment
open terminal on your environment
pip install --ignore-installed --upgrade tensorflow
conda install -c menpo opencv3
conda install -c conda-forge tqdm
pip install tflearn

train directory is training data
test directory is test data

catdog.py is training script, you will get test_dat.npy and train_daya.npy
result.py is to show result
