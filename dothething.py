import os

os.system('python train.py  --dataset_name acos/Restaurant-ACOS  --lr 7e-5 --n_epochs 50')
os.system('python train.py  --dataset_name acos/Restaurant-ACOS  --lr 5e-5 --n_epochs 50')
os.system('python train.py  --dataset_name acos/Restaurant-ACOS  --lr 3e-5 --n_epochs 50')
os.system('python train.py  --dataset_name acos/Restaurant-ACOS  --lr 1e-5 --n_epochs 50')
os.system('python train.py  --dataset_name acos/Restaurant-ACOS  --lr 2e-5 --n_epochs 50')
os.system('python train.py  --dataset_name acos/Restaurant-ACOS  --lr 4e-5 --n_epochs 50')
os.system('python train.py  --dataset_name acos/Restaurant-ACOS  --lr 6e-5 --n_epochs 50')

'''
for i in range(5):
    os.system('python train.py  --dataset_name penga/14res')    #1h21min
    print('done===========================-------------------')

for i in range(5):
    # os.system('python train.py  --dataset_name pengb/15res --batch_size 4')
    print('done===========================-------------------')
for i in range(5):
    os.system('python train.py  --dataset_name penga/15res ')
    print('done===========================-------------------')

for i in range(5):
    os.system('python train.py  --dataset_name penga/16res')
    print('done===========================-------------------')

for i in range(5):
    os.system('python train.py  --dataset_name penga/14lap  --lr 3e-5 --n_epochs 50')
    print('done===========================-------------------')
'''
""""
for i in range(2):
    os.system('python train.py  --dataset_name penga/14res')    #1h21min
    print('done===========================-------------------')

for i in range(5):
    #os.system('python train.py  --dataset_name pengb/15res ')
    print('done===========================-------------------')

for i in range(2):
    os.system('python train.py  --dataset_name penga/16res')
    print('done===========================-------------------')

for i in range(2):
    os.system('python train.py  --dataset_name penga/14lap  --lr 3e-5')
    print('done===========================-------------------')
"""