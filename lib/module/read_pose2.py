import cv2
import glob
import os

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import IPython

from lib.math import *
from lib import vision
from lib import util


def projection(T, intrinsic):
    w = intrinsic.w
    h = intrinsic.h
    fx = intrinsic.fx
    fy = intrinsic.fy
    cx = intrinsic.cx
    cy = intrinsic.cy

    X = T[0] 
    Y = T[1]
    Z = T[2]+1e-10
    
    u = w*(fx*(X/Z)+cx)
    v = h*(fy*(Y/Z)+cy)
    #v = h-h*(fy*(Y/Z)+cy)

    #v = -w*fx*(X/Z)  # w-w*(fx*(X/Z)+cx)
    #u = -h*fy*(Y/Z)  #h*(fy*(Y/Z)+cy)

    #v = -w*fx*(X/Z)  # w-w*(fx*(X/Z)+cx)
    #u = -h*fy*(Y/Z)  #h*(fy*(Y/Z)+cy)
    #u = np.clip(u,0,w)
    #v = np.clip(v,0,h)

    return [u,v]

def read(config): 
    data_name = config['data_name']
    nb_object = config['object']['nb_object']
    supervision = config['pose']['supervision']
    fps = config['animation']['fps']
    pose_path = './output/pose/'+data_name
    
    data_dir = './data/'+data_name
    output_dir = './output/read_pose2/'+data_name
    se3_dict = np.load(pose_path+'/se3_pose.npy').item()
    
    #
    g_rc = np.load(pose_path+'/g_rc.npy')[0,:]
    #g_rc = np.asarray([0.38457832, 0.09596953, -0.32798763, -0.34428167, 0.43178049, -0.17598158],dtype = np.float32)
    #g_rc = se3_to_SE3(g_rc)
    
    obj = vision.SE3object(np.zeros(6), angle_type = 'axis')
    intrinsic = vision.Zed_mini_intrinsic(scale = 2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for demo_name, se3_traj in se3_dict.items():
        output_path = output_dir+'/'+demo_name
        util.create_dir(output_path, clear = True)
        img_list = sorted(glob.glob(data_dir+'/'+demo_name+'/image/*.npy'))
        _, g_vr = util.load_vicon(data_dir+'/'+demo_name+'/camera_position0.npy')

        depth_dir = './data/'+data_name+'/'+demo_name+'/depth'
        mask_dir = './output/segment/'+data_name+'/'+demo_name
        depth_files = sorted(glob.glob(depth_dir+'/*.npy'))
        mask_files = sorted(glob.glob(mask_dir+'/*.npy'))
        
        obj_idx = 0
        ref_frame = 0
        se30 = se3_traj[ref_frame,obj_idx,:]
        mask0 = np.load(mask_files[ref_frame])
        depth0 = np.load(depth_files[ref_frame]) 
        init_R = se3_to_SE3(se30)[0:3,0:3]
        com = vision.get_com(mask0, depth0, obj_idx, init_R = init_R)

        #IPython.embed()
        g_vc = np.matmul(g_vr, g_rc)
        if (supervision == 'both_ends') or (supervision == 'full'):
            g_c_com = se3_to_SE3(com)
            g_vo = se3_to_SE3(se30)
            g_ov = inv_SE3(g_vo)
            g_cv = inv_SE3(g_vc)
            g_o_com = np.matmul( np.matmul(g_ov, g_vc), g_c_com) 
        elif supervision == 'never':
            g_c_com = se3_to_SE3(com)
            g_co = se3_to_SE3(se30)
            g_oc = inv_SE3(g_co)
            g_o_com = np.matmul(g_oc, g_c_com)
        #IPython.embed()
        for t in range(len(se3_traj)):
            ax.clear()
            img = np.load(img_list[t])
            img = cv2.resize(img, None, fx = 0.5, fy = 0.5)
            ax.imshow(img/255.)  
            for obj_idx in range(nb_object):
                ########################################
                if (supervision == 'both_ends') or (supervision == 'full'):
                    # g_co = g_rc^-1 * g_vr^-1 * g_vo = [g_vr*g_rc]^-1*g_vo
                    xi_vo = se3_traj[t,obj_idx,:]
                    g_vo = se3_to_SE3(xi_vo)
                    g_v_com = np.matmul(g_vo, g_o_com) #
                    g_c_com = np.matmul(g_cv, g_v_com)        
                    g_co = np.matmul(g_cv, g_vo)
                elif supervision == 'never':
                    xi_co = se3_traj[t,obj_idx,:]
                    g_co = se3_to_SE3(xi_co)
                    g_c_com = np.matmul(g_co, g_o_com)

                #'''
                if (supervision == 'both_ends') or (supervision == 'full'):
                    if t == 0:
                        g_c_com0 = np.copy(g_c_com)
                        g_vo0 = np.copy(g_vo)
                        SE3 = np.copy(g_c_com0)
                    else:
                        g_o0o1 = np.matmul(inv_SE3(g_vo0),g_vo)
                        SE3 = np.matmul(np.matmul(g_c_com0, np.matmul(inv_SE3(g_o_com), g_o0o1)), g_o_com)
                        g_c_com0 = np.copy(SE3)
                        g_vo0 = np.copy(g_vo)
                elif supervision == 'never':
                    SE3 = np.copy(g_c_com)
                #'''
                '''
                if supervision == 'both_ends' or 'full':
                    # g_co = g_rc^-1 * g_vr^-1 * g_vo = [g_vr*g_rc]^-1*g_vo
                    SE3 = np.matmul(np.matmul(inv_SE3(g_vc), g_vo),g_o_com)
                elif supervision == 'never':
                    SE3 = np.matmul(g_co, g_o_com)
                '''

                T = SE3[0:3,3]
                u,v = projection(T, intrinsic)
                ax.scatter(u,v, c = 'k')
                ########################################   
                se3 = SE3_to_se3(SE3)
                obj.apply_pose(se3)
                s = 0.1 *10

                scaled_xbasis = 0.1*(obj.xbasis-obj.orientation)+obj.orientation
                scaled_ybasis = 0.1*(obj.ybasis-obj.orientation)+obj.orientation
                scaled_zbasis = 0.1*(obj.zbasis-obj.orientation)+obj.orientation
                    
                x_u, x_v = projection(scaled_xbasis, intrinsic)
                y_u, y_v = projection(scaled_ybasis, intrinsic)
                z_u, z_v = projection(scaled_zbasis, intrinsic)

                #x_u, x_v = projection(obj.xbasis, intrinsic)
                #y_u, y_v = projection(obj.ybasis, intrinsic)
                #z_u, z_v = projection(obj.zbasis, intrinsic)

                
                #x_u_u = s* (x_u-u)/np.linalg.norm(x_u-u+1e-5) + u
                #x_v_v = s* (x_v-v)/np.linalg.norm(x_v-v+1e-5) + v
                #y_u_u = s* (y_u-u)/np.linalg.norm(y_u-u+1e-5) + u
                #y_v_v = s* (y_v-v)/np.linalg.norm(y_v-v+1e-5) + v
                #z_u_u = s* (z_u-u)/np.linalg.norm(z_u-u+1e-5) + u
                #z_v_v = s* (z_v-v)/np.linalg.norm(z_v-v+1e-5) + v
                
                
                '''     
                x_u = s*np.clip(x_u-u,2e-1, 2) + u 
                x_v = s*np.clip(x_v-v,2e-1, 2) + v
                y_u = s*np.clip(y_u-u,2e-1, 2) + u
                y_v = s*np.clip(y_v-v,2e-1, 2) + v
                z_u = s*np.clip(z_u-u,2e-1, 2) + u
                z_v = s*np.clip(z_v-v,2e-1, 2) + v
                '''
                
                #'''
                x_u_u = s*(x_u-u) + u
                x_v_v = s*(x_v-v) + v
                y_u_u = s*(y_u-u) + u
                y_v_v = s*(y_v-v) + v
                z_u_u = s*(z_u-u) + u
                z_v_v = s*(z_v-v) + v
                #'''

                ax.plot([u, x_u_u], [v, x_v_v], c = 'r', linewidth = 3)
                ax.plot([u, y_u_u], [v, y_v_v], c = 'g', linewidth = 3)
                ax.plot([u, z_u_u], [v, z_v_v], c = 'b', linewidth = 3)
                ########################################
                ax.set_xlim([0, intrinsic.w])
                ax.set_ylim([intrinsic.h,0])
                #IPython.embed()
                #break
            fig.savefig(output_path+'/%06d.png'%t)
        video_path = output_path+'/se3_on_image.avi'
        util.frame_to_video(output_path, video_path, 'png', fps=fps)()

