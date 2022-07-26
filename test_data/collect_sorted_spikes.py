import tables
import numpy as np
from tqdm import tqdm
import os
from glob import glob
from joblib import Parallel, delayed, cpu_count
import re
from shutil import copy as cp
import pandas as pd

PATH_FILE='/media/bigdata/Abuzar_Data/all_h5_files.txt'
sorted_save_path = '/media/bigdata/neuRecommend/test_data/sorted'
raw_save_path = '/media/bigdata/neuRecommend/test_data/raw'

path_list = [x.strip() for x in open(PATH_FILE,'r').readlines()]
dir_list = [os.path.dirname(x) for x in path_list]
basename_list = [os.path.basename(x).split('.')[0] for x in path_list]

############################################################
# Find number of sorted waveforms
############################################################
#shape_list = []
for name, this_path in tqdm(zip(basename_list, path_list)):
    #this_path = path_list[0]
    with tables.open_file(this_path,'r') as h5:
        #h5 = tables.open_file(this_path,'r')
        #this_shapes = np.stack(
        #        [x['waveforms'].shape for x in h5.iter_nodes('/sorted_units')])
        #shape_list.append(this_shapes)
        for num, val in enumerate(h5.iter_nodes('/sorted_units')):
            np.save(os.path.join(sorted_save_path, name + f"_unit{num}"),
                    val['waveforms'][:])

#shape_array = np.concatenate(shape_list,axis=0)

############################################################
# Find total number of putative spikes extracted
############################################################
waveform_file_path = '/media/bigdata/Abuzar_Data/all_spike_waveform_files.txt'
waveform_file_list = [x.strip() for x in open(waveform_file_path,'r').readlines()]
waveform_basename_list = [x.split('/')[-4] for x in waveform_file_list]
waveform_elec_list = [re.findall('\d+',x.split('/')[-2])[0] \
                            for x in waveform_file_list]
dest_list = [os.path.join(raw_save_path, x + '_elec' + y + '.npy') \
                    for x,y in zip(waveform_basename_list, waveform_elec_list)]

src_dest_frame = pd.DataFrame({
                    'src' : waveform_file_path,
                    'dest' : dest_list})

#total_waveform_shape_list = []
#for this_path in tqdm(waveform_file_list):
#    #this_path = waveform_file_list[0]
#    total_waveform_shape_list.append(np.load(this_path).shape)

def return_shape(this_path):
    return np.load(this_path).shape

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def copy_file(x):
    cp(*x)

outs = parallelize(copy_file, zip(waveform_file_list, dest_list))

#outs = parallelize(return_shape, waveform_file_list)
#waveform_shape_array = np.stack(outs)

########################################
# Set aside not unit waveforms
########################################
marked_file_path = '/media/bigdata/Abuzar_Data/all_spike_waveform_files_marked.csv'
marked_frame = pd.read_csv(marked_file_path, header=None)
marked_frame.dropna(inplace=True)

neg_save_dir = '/media/bigdata/neuRecommend/test_data/sorted/neg'

neg_file_list = marked_frame.iloc[:,0]
neg_basename_list = [x.split('/')[-4] for x in neg_file_list]
neg_elec_list = [re.findall('\d+',x.split('/')[-2])[0] \
                            for x in neg_file_list]
neg_dest_list = [os.path.join(neg_save_dir, x + '_elec' + y + '.npy') \
                    for x,y in zip(neg_basename_list, neg_elec_list)]
outs = parallelize(copy_file, zip(neg_file_list, neg_dest_list))

########################################
## Aggregate waveforms into HDF5 
########################################
hf5 = tables.open_file(os.path.join(sorted_save_path, 'sorted_waveforms.h5')
                    , 'w', title = 'sorted_waveforms')
hf5.create_group('/sorted', 'pos', createparents=True)
hf5.create_group('/sorted', 'neg', createparents=True)

file_list = glob(os.path.join(sorted_save_path, "*", "*.npy"))

for this_file in tqdm(file_list):
    x = np.load(this_file) 
    basename = this_file.split('/')[-1].split('.')[0]
    wave_class = this_file.split('/')[-2]
    if wave_class == 'pos':
        save_path = '/sorted/pos'
    else:
        save_path = '/sorted/neg'
    if not os.path.join(save_path, basename) in hf5:
        hf5.create_array(save_path,basename,x)

hf5.flush()
hf5.close()
