import os
import re
import numpy as np
import pydicom
import torch
import torchvision
import argparse
from tqdm import tqdm
from warnings import warn
from skimage.morphology import closing,disk
from skimage.exposure import equalize_adapthist

parser = argparse.ArgumentParser(
                    prog='Gold Atlas Dicom Parser',
                    description='Parse dicom directories into numpy format.')
parser.add_argument('source',
                    help='Parent directory for all patient folders.')
parser.add_argument('out',default='out',
                    help='Directory to pass all outputs to. By default will create directory in WD called `out\'.')
args = parser.parse_args()


target_res = 512

use_loc1 = True
use_loc2 = True
use_loc3 = True


drop_names = ["2_10_P"]

out_dir = args.out
if not os.path.exists(out_dir):
    print(f'No output directory {out_dir}. Creating it...')
    os.makedirs(out_dir)
parent_dir = args.source

names_l1=["1_01_P",
        "1_02_P",
        "1_03_P",
        "1_04_P",
        "1_05_P",
        "1_06_P",
        "1_07_P",
        "1_08_P"] if use_loc1 else []
names_l2=["2_03_P",
        "2_04_P",
        "2_05_P",
        "2_06_P",
        "2_09_P",
        "2_10_P",
        "2_11_P"] if use_loc2 else []
names_l3=["3_01_P",
        "3_02_P",
        "3_03_P",
        "3_04_P"] if use_loc3 else []

train_pats = names_l1[:5] + names_l2[:4] + names_l3[:2]
val_pats = names_l1[5:6] + names_l2[5:6] + names_l3[2:3]
test_pats = names_l1[6:] + names_l2[6:] + names_l3[3:]

t0 = all([a not in val_pats for a in train_pats])
t1 = all([a not in test_pats for a in train_pats])
t2 = all([a not in test_pats for a in val_pats])

assert t0, "Train/validation crossover"
assert t1, "Train/test crossover"
assert t2, "validation/test crossover"


train_paths = [os.path.join(parent_dir, a) for a in train_pats]
val_paths = [os.path.join(parent_dir, a) for a in val_pats]
test_paths = [os.path.join(parent_dir, a) for a in test_pats]

#List where CT is reversed compared to MRI
rev_list = [
        "2_03_P",
        "2_04_P",
        "2_05_P",
        "2_06_P",
        "2_09_P",
        "2_10_P",
        "2_11_P"
        ]


def build_dict_from_paths(paths):
    out = {}
    get_loc_ct = re.compile(".*\.(\d+)\.dcm$")
    for c_path in paths:
        c_files = [os.path.join(c_path,a) for a in os.listdir(c_path)]
        c_files = [a for a in c_files if os.path.isfile(a)]
        c_files = [a for a in c_files if a[-4:]==".dcm"]
        for c_file in c_files:
            c_dcmf = pydicom.dcmread(c_file)
            mode = c_dcmf.Modality
            if mode in ["CT","MR"]:
                p_id = c_dcmf.PatientID
                sdes = c_dcmf.SeriesDescription
                if mode == "CT":
                    if "CTtoMR" in sdes:
                        loc = int(get_loc_ct.match(c_file).group(1))
                        if out.get(p_id) is None:
                            out[p_id] = {}
                        if out[p_id].get(loc) is None:
                            out[p_id][loc] = {}
                        if out[p_id][loc].get("CT") is not None:
                            raise Exception(f"Double on CT for patient {p_id} at location {loc}")
                        out[p_id][loc]["CT"] = c_file
                else: #Is MR
                    if "T2" in sdes or "t2" in sdes:
                        loc = int(c_dcmf.InstanceNumber)
                        if out.get(p_id) is None:
                            out[p_id] = {}
                        if out[p_id].get(loc) is None:
                            out[p_id][loc] = {}
                        if out[p_id][loc].get("MR") is not None:
                            raise Exception(f"Double on MR for patient {p_id} at location {loc}")
                        out[p_id][loc]["MR"] = c_file
    return out

def reverse_ct_order(p_dict):
    locs = list(p_dict.keys())
    locs = list(sorted(locs))
    c_order = [p_dict[loc]["CT"] for loc in locs]
    r_order = list(reversed(c_order))
    for i,loc in enumerate(locs):
        p_dict[loc]["CT"] = r_order[i]
    return p_dict

def fix_reversed_orders(c_dict):
    for pid in c_dict:
        if pid in rev_list:
            print(f"Reversing order for {pid}")
            c_dict[pid] = reverse_ct_order(c_dict[pid])
    return c_dict

def get_mask(ctimg):
    val_thres = -500 #Value threshold
    ver_thres = 440 #Vertical threshold
    dis_size = 15 #Closing footprint size
    mask = ctimg[0].copy()
    mask[ctimg[0]<=val_thres] = 0
    mask[ctimg[0]>val_thres] = 1
    mask = closing(mask.astype(np.uint8),disk(dis_size))
    mask[ver_thres:] = 0
    mask = mask.astype(bool)
    return mask

def save_as_npy_dir(out_dir,t_dict):
    for pid in tqdm(t_dict):
        if pid in drop_names:
            print(f"Dropping {pid}")
            continue
        for loc in t_dict[pid]:
            cmask = None
            pc = os.path.join(out_dir,"CT", f"{pid}_{loc}.npy")
            pm = os.path.join(out_dir,"MR", f"{pid}_{loc}.npy")
            if os.path.exists(pc) and os.path.exists(pm):
                warn(f"File found for {pid} at {loc}")
                continue
            for mode in ["CT","MR"]:
                #Get paths
                s_path = t_dict[pid][loc][mode]
                t_path = os.path.join(out_dir,mode, f"{pid}_{loc}.npy")

                #Read and process data
                c_dcmf = pydicom.dcmread(s_path)
                arr = c_dcmf.pixel_array
                arr = pydicom.pixel_data_handlers.util.apply_modality_lut(arr,c_dcmf)
                arr = np.expand_dims(arr,0)
                arr = arr.astype('float')
                if arr.shape[1] != target_res:
                    arr = torch.from_numpy(arr)
                    arr = torchvision.transforms.functional.resize(arr,(target_res,target_res)).numpy()
                
                if mode == "CT":
                    cmask = get_mask(arr)
                    cmask = np.expand_dims(cmask,0)
                
                #normalize
                arr = arr - np.min(arr)
                if np.max(arr) != 0:
                    arr = arr / np.max(arr)

                #Mask
                arr[np.invert(cmask)] = 0

                np.save(t_path,arr)

train_dir = os.path.join(out_dir,"train")
val_dir = os.path.join(out_dir,"val")
test_dir = os.path.join(out_dir,"test")

#Make sure directories exist
for cd in [train_dir,val_dir,test_dir]:
    if not os.path.exists(cd):
        os.makedirs(cd)
    ctd = os.path.join(cd,'CT')
    if not os.path.exists(ctd):
        os.makedirs(ctd)
    mrd = os.path.join(cd,'MR')
    if not os.path.exists(mrd):
        os.makedirs(mrd)


print("Parsing directories...")
train_dict = build_dict_from_paths(train_paths)
train_dict = fix_reversed_orders(train_dict)
val_dict = build_dict_from_paths(val_paths)
val_dict = fix_reversed_orders(val_dict)
test_dict = build_dict_from_paths(test_paths)
test_dict = fix_reversed_orders(test_dict)

print("Saving npy...")
save_as_npy_dir(train_dir,train_dict)
save_as_npy_dir(val_dir,val_dict)
save_as_npy_dir(test_dir,test_dict)

print("Done!")

