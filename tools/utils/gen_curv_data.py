#!/usr/bin/env python3
# Developed by Junyi Ma

try: from utils import *
except: from utils import *


def gen_curv_data(depth_folder, dst_folder, normalize=False):
  """ Generate projected range data in the shape of (64, 900, 1).
      The input raw data are in the shape of (Num_points, 3).
  """
  # specify the goal folder
  dst_folder = os.path.join(dst_folder, 'curvature')
  try:
    os.stat(dst_folder)
    print('generating curvature data in: ', dst_folder)
  except:
    print('creating new curvature folder: ', dst_folder)
    os.mkdir(dst_folder)
  
  # load LiDAR scan files
  depth_paths = load_files(depth_folder)
#   print(depth_paths)
  curv_list = []
  # iterate over all scan files
  for idx in range(len(depth_paths)):
    #   print(idx)
    # 先读取一张图，进行点线处理
    # print(depth_paths[idx])
    img_depth = np.load(depth_paths[idx])
    # print(img_depth.shape)  # (64, 900)

    # 将-1全改为0
    img_depth[img_depth == -1] = 0
    # print(img_depth.shape)

    # 计算平滑度：
    row_curvature = np.ones_like(img_depth)
    row_curvature = - row_curvature
    for i in range(img_depth.shape[0]):
        row_value = img_depth[i, :]
        start_ind = 5
        for j in range(img_depth.shape[1]-10):
            cur_ind = start_ind + j
            if (row_value[cur_ind]!=0):
                diff_range = row_value[cur_ind-5] + row_value[cur_ind-4] + row_value[cur_ind-3] + row_value[cur_ind-2] + row_value[cur_ind-1] \
                            - row_value[cur_ind] * 10  \
                            + row_value[cur_ind+5] + row_value[cur_ind+4] + row_value[cur_ind+3] + row_value[cur_ind+2] + row_value[cur_ind+1]
                row_curvature[i,cur_ind] = diff_range   
            # else:
            #     row_curvature[i,cur_ind] = -1
    if normalize:
      row_curvature = row_curvature / np.max(row_curvature)
    dst_path = os.path.join(dst_folder, str(idx).zfill(6))
    np.save(dst_path, row_curvature)
    print('finished generating curvature data at: ', dst_path)
    curv_list.append(row_curvature)


    
    # proj_range, _, _, _ = range_projection(current_vertex)  # proj_ranges   from larger to smaller

    
    # normalize the image
    # if normalize:
    #   proj_range = proj_range / np.max(proj_range)
    
    # # generate the destination path
    # dst_path = os.path.join(dst_folder, str(idx).zfill(6))
    
    # # save the semantic image as format of .npy
    # np.save(dst_path, proj_range)
    # depths.append(proj_range)
    # print('finished generating depth data at: ', dst_path)

  return curv_list
  

if __name__ == '__main__':
  depth_folder = '/home/mjy/repo/OverlapNet_for_TF2/data/preprocess_data_demo/depth/'
  dst_folder = '/home/mjy/repo/OverlapNet_for_TF2/data/preprocess_data_demo'
  
  depth_data = gen_curv_data(depth_folder, dst_folder)
