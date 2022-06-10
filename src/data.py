import os
import random
import shutil
import glob

# Organize data into train, validation, and test directory
os.chdir('data')
if (os.path.isdir('train/normal')) & (os.path.isdir('train/tuberculosis')) is False:
    os.makedirs('train/normal')
    os.makedirs('train/tuberculosis')

if (os.path.isdir('validation/normal')) & (os.path.isdir('validation/tuberculosis')) is False:
    os.makedirs('validation/normal')
    os.makedirs('validation/tuberculosis')

if (os.path.isdir('testing/normal')) & (os.path.isdir('testing/tuberculosis')) is False:
    os.makedirs('testing/normal')
    os.makedirs('testing/tuberculosis')

for i in random.sample(glob.glob('Normal*'), 2240):
    shutil.move(i, 'train/normal')
for i in random.sample(glob.glob('Tuberculosis*'), 2240):
    shutil.move(i, 'train/tuberculosis')
for i in random.sample(glob.glob('Normal*'), 560):
    shutil.move(i, 'validation/normal')
for i in random.sample(glob.glob('Tuberculosis*'), 560):
    shutil.move(i, 'validation/tuberculosis')
for i in random.sample(glob.glob('Normal*'), 700):
    shutil.move(i, 'testing/normal')
for i in random.sample(glob.glob('Tuberculosis*'), 700):
    shutil.move(i, 'testing/tuberculosis')
os.chdir('../')

base_dir = "data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

# Directory with training normal/tb pictures
train_normal_dir = os.path.join(train_dir, 'normal')
train_tb_dir = os.path.join(train_dir, 'tb')

# Directory with validation normal/tb pictures
val_normal_dir = os.path.join(val_dir, 'normal')
val_tb_dir = os.path.join(val_dir, 'tb')