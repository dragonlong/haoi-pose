# examples/Python/Advanced/global_registration.py

import open3d as o3d
import numpy as np
import copy
from glob import glob

def bp():
    import pdb;pdb.set_trace()

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source_name, target_name, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(source_name, format='xyz')
    target = o3d.io.read_point_cloud(target_name, format='xyz')
    # source = o3d.io.read_point_cloud("./assets/cloud_bin_0.pcd")
    # target = o3d.io.read_point_cloud("./assets/cloud_bin_1.pcd")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    # result = o3d.pipelines.registration.registration_icp(
    #     source, target, distance_threshold, result_ransac.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    result =  o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return result

# def get_gt_data():
#     res_path = f'{my_dir}/results/test_pred/oracle/{cfg.exp_num}_unseen_part_rt_pn_general.npy'
#     results  = np.load(res_path, allow_pickle=True).item()
#     infos_dict, track_dict = results['info'], results['err']
#     basenames = infos_dict['basename']
#     inputs   = np.array(infos_dict['in'])
#     r_gt     = np.array(infos_dict['r_gt'])
#     t_gt     = np.array(infos_dict['t_gt'])
#     r_pred   = np.array(infos_dict['r_pred'])
#     rdiff    = np.array(track_dict['rdiff'])

if __name__ == "__main__":
    voxel_size = 0.02  # means 5cm for the dataset
    # 0.8475 # 0.855 #0.862 # 0.851 # 0.845 # 0.81 # 0.84
    exp_num    = '0.81'
    query_keys = ['canon', 'target', 'input', 'pred']
    fpath   = f'/home/dragon/Documents/ICML2021/results/preds/{exp_num}/generation/' # generation/generation
    fnames  = {}
    for key in query_keys:
        fnames[key] = sorted(glob(f'{fpath}/*{key}*txt'))
    for fn in fnames['canon']:
        source_name = fn
        target_name = fn.replace('canon', 'input')

        source, target, source_down, target_down, source_fpfh, target_fpfh = \
                prepare_dataset(source_name, target_name, voxel_size)
        print('1. ', np.asarray(target.points).mean())
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        print(result_ransac.transformation)
        draw_registration_result(source_down, target_down,
                                 result_ransac.transformation)

        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                         voxel_size)
        print('2. ', np.asarray(target.points).mean())
        print(result_icp.transformation)
        draw_registration_result(source, target, result_icp.transformation)
        new_pcd = copy.deepcopy(source).transform(result_icp.transformation)
        transformed_pts = np.asarray(new_pcd.points)
        np.savetxt(fn.replace('canon', 'icp'), transformed_pts)
        print('saving to', fn.replace('canon', 'icp'))
