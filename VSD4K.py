from PIL import Image
import numpy as np
import argparse
import os
import math
import copy
import torchvision.transforms as transforms
global fold_name

parser = argparse.ArgumentParser()

#######################path to your datasets########################
parser.add_argument("--source_path",type=str,default="/home/lee/data")

parser.add_argument("--size_w",type=int,default=20, help="number of patches on wide")
parser.add_argument("--size_h",type=int,default=10, help="number of patches on height")
parser.add_argument("--type", type=int, help="SR scale", default=2, choices=[2,3,4])
parser.add_argument("--tt",type= str, help="video topic and video length",default='None')
parser.add_argument("--k",type=int,default=2,help='number of clusters')
parser.add_argument("--time",type=int,default=15,help='video seconds')
parser.add_argument("--pre_pro",action='store_true',default=False)
parser.add_argument("--model",type= str, help="model type",default='')
args, _ = parser.parse_known_args()
params = parser.parse_args()
size_w = params.size_w
size_h = params.size_h
h,w = 1080,1920
len_h = int(h / size_h)
len_w = int(w / size_w)
# Default FPS is 30
num_frames = params.time*30
num_patches = num_frames*size_h*size_w

def img_patches(path,f_num,size_w,size_h,folder):
    """
    :param path: frame path
    :param size_w: patch size wide
    :param size_h: patch size height
    """
    image = Image.open(path)
    ima = np.array(image)
    box_list = []
    h = ima.shape[0]
    w = ima.shape[1]
    len_h = int(h/size_h)
    len_w = int(w/size_w)
    count = 0
    num_patches = size_w*size_h
    for i in range(0,size_h):
        for j in range(0,size_w):
            count = count+1
            box = (j*len_w,i*len_h,(j+1)*len_w,(i+1)*len_h,)
            box_list.append(box)
    patch_list = [image.crop(box) for box in box_list]
    index = 1
    for img in patch_list:
        img.save(folder+'/'+str(index+f_num*num_patches)+'.png')
        index = index + 1

def frame_check(frame):
    for i in frame:
        if i.__contains__('.png'):
            pass
        else:
            frame.remove(i)
    return frame
def frame_reorg(frame,num_frames):
    return frame[0:num_frames]

def generate_test_data(dir_path,data_type):
    global frame
    cur_path = params.source_path
    cur_path = os.path.join(cur_path, params.tt)
    if data_type =='lr':
        fold_name = 'data_divide/'+'DIV2K_train_LR_bicubic_0/'+'X'+str(params.type)
        folder = os.path.join(cur_path, fold_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        frame = os.listdir(dir_path)
        frame = frame_check(frame)
        frame.sort(key = lambda x:int(x[:-6]))
    if data_type =='hr':
        fold_name = 'data_divide/'+'DIV2K_train_HR_0/'+'X'+str(params.type)
        folder = os.path.join(cur_path, fold_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        frame = os.listdir(dir_path)
        frame = frame_check(frame)
        frame.sort(key = lambda x:int(x[:-4]))
    frame = frame[0:num_frames]
    frame_path = []
    index = 0
    for frame_name in frame:
        frame_path.append(dir_path + '/' + frame_name)
    for path in frame_path:
        if path.__contains__('.png'):
            f_num =index
            img_patches(path,f_num,size_w,size_h,folder)
            index = index + 1
    return fold_name

def get_label():
    os.system('cd script/'+params.tt+'&& bash pre_process'+'X'+str(params.type))

def img_cluster(k, fold_name,data_type):
    cur_path = params.source_path
    cur_path = os.path.join(cur_path, params.tt)
    folder = os.path.join(cur_path, fold_name)
    labels = np.load('label_'+params.tt+'.npy')
    b = copy.deepcopy(labels)
    b.sort()
    k_point = [0]
    for i in range(0, k):
        p = int((i + 1) * labels.shape[0] / k)
        k_point.append(b[p - 1])
    patch_path = []
    patches = os.listdir(folder)
    patches.sort(key = lambda x:int(x[:-4]))
    for patch_name in patches:
        patch_path.append(folder+'/'+patch_name)
    for i in range(0, k):
        if data_type == 'lr':
            if not os.path.exists(cur_path+'/DIV2K_train_LR_bicubic'+'_chunk' + str(i)+'/X'+str(params.type)):
                os.makedirs(cur_path+'/DIV2K_train_LR_bicubic'+'_chunk' + str(i)+'/X'+str(params.type))
        else:
            if not os.path.exists(cur_path+'/DIV2K_train_HR'+'_chunk' + str(i)+'/X'+str(params.type)):
                os.makedirs(cur_path+'/DIV2K_train_HR'+'_chunk' + str(i)+'/X'+str(params.type))
    if data_type == 'lr':
        print("processing lr img")
        count = 0
        for i in labels:
            for j in range(0, len(k_point) - 1):
                l = j + 1
                if i > k_point[j] and i <= k_point[l]:
                    image = Image.open(patch_path[count])
                    image.save(patch_path[count].replace(cur_path+'/'+fold_name+'/', cur_path+'/DIV2K_train_LR_bicubic'+'_chunk'+str(j)+'/'+'X'+str(params.type)+'/'))
            count = count + 1
            if count%10000 == 0:
                print(str(count)+"/"+str(size_h*size_w*num_frames))

    if data_type == 'hr':
        print("processing hr img")
        count = 0
        for i in labels:
            for j in range(0, len(k_point) - 1):
                l = j + 1
                if i > k_point[j] and i <= k_point[l]:
                    image = Image.open(patch_path[count])
                    image.save(patch_path[count].replace(cur_path+'/'+fold_name+'/', cur_path+'/DIV2K_train_HR'+'_chunk'+str(j)+'/'+'X'+str(params.type)+'/'))
            count = count + 1
            if count % 10000 == 0:
                print(str(count) + "/" + str(size_h * size_w * num_frames))


def img_compose(file_path,num_frames):
    gen_img = Image.new('RGB', (1920, 1080), 'black')
    count = 0
    cur_path = params.source_path + '/' +params.tt
    if params.baseline is not None:
        fold_name = 'output'+params.baseline+'_X'+str(params.type)
    else:
        fold_name = 'output'+'_X'+str(params.type)
    folder = os.path.join(cur_path, fold_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    for f in range(0,num_frames):
        for i in range(0, size_h):
            for j in range(0, size_w):
                sub_img = Image.open(file_path[count][1])
                gen_img.paste(sub_img, (j * len_w, i * len_h, (j + 1) * len_w, (i + 1) * len_h,))
                count = count + 1
        gen_img.save(folder+'/'+str(f+1)+'.png')
    return folder

def get_patch_path(k,root_path):
    file_path = []
    chunk = '/chunk'
    print('fetch data from time&space chunk')
    for i in range(0,k):
        path = root_path+chunk+str(i)+'_'+params.model+'_'+'X'+str(params.type)
        patches = os.listdir(path)
        patches.sort(key=lambda x: int(x[:-4]))
        for f in patches:
            if f.endswith('.png'):
                file_path.append((f, os.path.join(path, f)))
    return file_path


def get_patch_psnr(sr,hr,shave=4):
    mse = 0
    l = len(hr)
    for i in range(0,l):
        sr_image = np.asarray(Image.open(sr[i][1]))
        sr_image = transforms.functional.to_tensor(sr_image)
        hr_image = np.asarray(Image.open(hr[i][1]))
        hr_image = transforms.functional.to_tensor(hr_image)
        sr_image = sr_image.to(hr_image.dtype)
        sr_image = (sr_image * 255).round().clamp(0, 255) / 255
        diff = sr_image - hr_image
        if shave:
            diff = diff[..., shave:-shave, shave:-shave]
        mse = diff.pow(2).mean([-3, -2, -1]) + mse
    mse = mse/l
    psnr = -10 * mse.log10()
    return psnr

def get_hr_patches_path(tt,type,k):
    root_path = params.source_path
    file_path = []
    tt = '/'+tt
    # chunk_count = [0]
    chunk = 'chunk'
    for i in range(0, k):
        path = root_path + tt + '/DIV2K_train_HR_'+chunk+str(i)+'/' + 'X' + str(type)
        patches = os.listdir(path)
        patches.sort(key=lambda x: int(x[:-4]))
        # chunk_count =chunk_count.append( len(patches))
        for f in patches:
            if f.endswith('.png'):
                file_path.append((f, os.path.join(path, f)))
    return file_path

def psnr(sr, hr, shave=4):
  sr = sr.to(hr.dtype)
  sr = (sr * 255).round().clamp(0, 255) / 255
  diff = sr - hr
  if shave:
    diff = diff[..., shave:-shave, shave:-shave]
  mse = diff.pow(2).mean([-3, -2, -1])
  psnr = -10 * mse.log10()
  return psnr.mean()

if __name__ == "__main__":
    if params.pre_pro:
        cur_path = params.source_path
        cur_path = os.path.join(cur_path, params.tt)
        lr_path = cur_path+'/DIV2K_train_LR_bicubic/'+'X'+str(params.type)
        hr_path = cur_path+'/DIV2K_train_HR'
        #split images into patches
        f = generate_test_data(lr_path,'lr')
        f1 = generate_test_data(hr_path,'hr')
        get_label()
        img_cluster(params.k,'data_divide/'+'DIV2K_train_LR_bicubic_0/'+'X'+str(params.type),'lr')
        img_cluster(params.k,'data_divide/'+'DIV2K_train_HR_0/'+'X'+str(params.type),'hr')
    else:
        cur_path = params.source_path
        cur_path = os.path.join(cur_path, params.tt)
        path = cur_path + '/save_img_fold'
        ff = get_patch_path(params.k, path)
        ff_hr = get_hr_patches_path(params.tt,params.type,params.k)
        psnr_all = get_patch_psnr(ff,ff_hr)
        print('all chunks avg PSNR:', round(psnr_all.item(), 4))


