import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import IPython

from lib.math import *
from lib import vision
from lib import util


def read(config):
    #read_pose(config)
    read_transformed_pose(config)
    #read_com(config)
    #read_pose_com(config)

def read_pose(config):
    data_name = config['data_name']
    nb_object = config['object']['nb_object']
    se3_path = './output/pose/'+data_name+'/se3_pose.npy'
    demo_list = os.listdir('./data/'+data_name)
    output_dir = './output/read_pose/'+data_name
    
    se3_dict = np.load(se3_path).item()
    obj = vision.SE3object(np.zeros(6), angle_type = 'axis')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    color = ['g','r']
    # draw trajectory w.r.t. camera
    for demo_name, se3_traj in se3_dict.items():
        output_path = output_dir+'/'+demo_name
        util.create_dir(output_path, clear = True)

        for t in range(len(se3_traj)):
            ax.clear()
            for obj_idx in range(nb_object):

                ax.plot(se3_traj[:t+1,obj_idx,0], se3_traj[:t+1,obj_idx,1], se3_traj[:t+1,obj_idx,2],
                        color[obj_idx], alpha = 0.5, linewidth = 4)
                obj.apply_pose(se3_traj[t,obj_idx,:])
                obj.plot(ax, scale = 0.05, linewidth = 3)


                #ax.set_aspect('equal')
            ax.axis('scaled')
            ax.set_xlabel('x(m)')
            ax.set_ylabel('y(m)')
            ax.set_zlabel('z(m)')
            ax.view_init(elev = -60, azim = -90)
            fig.savefig(output_path+'/%06d.png'%t)

        video_path = output_path+'/object_trajectory.avi'
        util.frame_to_video(output_path, video_path, 'png', fps=20)()

    #IPython.embed()
    

def read_transformed_pose(config):
    data_name = config['data_name']
    nb_object = config['object']['nb_object']
    fps = config['animation']['fps']

    se3_path = './output/pose/'+data_name+'/se3_pose.npy'
    demo_list = os.listdir('./data/'+data_name)
    output_dir = './output/read_pose/'+data_name
    
    se3_dict = np.load(se3_path).item()
    obj = vision.SE3object(np.zeros(6), angle_type = 'axis')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    color = ['g','r']
    # draw trajectory w.r.t. camera
    for demo_name, se3_traj in se3_dict.items():
        depth_dir = './data/'+data_name+'/'+demo_name+'/depth'
        mask_dir = './output/segment/'+data_name+'/'+demo_name
        depth_files = sorted(glob.glob(depth_dir+'/*.npy'))
        mask_files = sorted(glob.glob(mask_dir+'/*.npy'))
        output_path = output_dir+'/'+demo_name
        util.create_dir(output_path, clear = True)

        depth = np.load(depth_files[0])
        mask = np.load(mask_files[0])  

        se30 = se3_traj[0,0,:]
        SE30 = se3_to_SE3(se30)
        T0 = SE30[0:3,3]
        T_traj = np.expand_dims(T0,0)
        # modify for multiple object
        for t in range(len(se3_traj)):
            ax.clear()
            for obj_idx in range(nb_object):
                se3 = se3_traj[t,obj_idx,:]
                SE3 = se3_to_SE3(se3)
                T = SE3[0:3,3]
                T_traj = np.concatenate([T_traj, np.expand_dims(T,0)], 0)
                ax.plot(T_traj[:,0], T_traj[:,1], T_traj[:,2],
                        color[obj_idx], alpha = 0.5, linewidth = 4)
                obj.apply_pose(se3)
                obj.plot(ax, scale = 0.05, linewidth = 3) 

            util.set_axes_equal(ax)
            #ax.axis('scaled')
            ax.set_xlabel('x(m)')
            ax.set_ylabel('y(m)')
            ax.set_zlabel('z(m)')
            ax.view_init(elev = -60, azim = -90)
            fig.savefig(output_path+'/%06d.png'%t)

        np.save(output_path+'/pose_traj.npy', se3_traj)
        video_path = output_path+'/object_trajectory.avi'
        util.frame_to_video(output_path, video_path, 'png', fps=fps)()


def read_com(config):
    data_name = config['data_name']
    nb_object = config['object']['nb_object']
    demo_list = os.listdir('./data/'+data_name)
    output_dir = './output/read_pose/'+data_name
    com_dict = {}

    for demo_name in demo_list:
        depth_dir = './data/'+data_name+'/'+demo_name+'/depth'
        mask_dir = './output/segment/'+data_name+'/'+demo_name
        depth_files = sorted(glob.glob(depth_dir+'/*.npy'))
        mask_files = sorted(glob.glob(mask_dir+'/*.npy'))

        com_traj = []
        for depth_file, mask_file in zip(depth_files, mask_files):
            depth = np.load(depth_file) 
            mask = np.load(mask_file)  
            
            com_one = []
            for obj_idx in range(nb_object):
                com = vision.get_com(mask, depth, obj_idx)
                com_one.append(np.expand_dims(com, axis = 0))
            com_one = np.concatenate(com_one, axis = 0)
            com_traj.append(np.expand_dims(com_one, axis = 0))
        com_traj = np.concatenate(com_traj,0)
        com_dict[demo_name] = com_traj
    #IPython.embed()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    obj = vision.SE3object(np.zeros(6), angle_type = 'axis')
    for demo_name, com_traj in com_dict.items():
        output_path = output_dir+'/'+demo_name
        for t in range(len(com_traj)):
            ax.clear()
            for obj_idx in range(nb_object):
                ax.plot(com_traj[:t+1,obj_idx,0], com_traj[:t+1,obj_idx,1], com_traj[:t+1,obj_idx,2])
                obj.apply_pose(com_traj[t,obj_idx,:])
                obj.plot(ax, scale = 0.1, linewidth = 3)

            ax.axis('scaled')
            ax.set_xlabel('x(m)')
            ax.set_ylabel('y(m)')
            ax.set_zlabel('z(m)')
            ax.view_init(elev = -60, azim = -90)
            fig.savefig(output_path+'/com_%06d.png'%t)


def read_pose_com(config):
    data_name = config['data_name']
    nb_object = config['object']['nb_object']
    se3_path = './output/pose/'+data_name+'/se3_pose.npy'
    demo_list = os.listdir('./data/'+data_name)
    output_dir = './output/read_pose/'+data_name
    
    se3_dict = np.load(se3_path).item()
    com_dict = {}
    for demo_name in demo_list:
        depth_dir = './data/'+data_name+'/'+demo_name+'/depth'
        mask_dir = './output/segment/'+data_name+'/'+demo_name
        depth_files = sorted(glob.glob(depth_dir+'/*.npy'))
        mask_files = sorted(glob.glob(mask_dir+'/*.npy'))

        com_traj = []
        for depth_file, mask_file in zip(depth_files, mask_files):
            depth = np.load(depth_file) 
            mask = np.load(mask_file)  
            
            com_one = []
            for obj_idx in range(nb_object):
                com = vision.get_com(mask, depth, obj_idx)
                com_one.append(np.expand_dims(com, axis = 0))
            com_one = np.concatenate(com_one, axis = 0)
            com_traj.append(np.expand_dims(com_one, axis = 0))
        com_traj = np.concatenate(com_traj,0)
        com_dict[demo_name] = com_traj
    
    obj = vision.SE3object(np.zeros(6), angle_type = 'axis')
    fig = plt.figure()
    color = ['g','r']
    ax = fig.add_subplot(111, projection='3d')

    # draw trajectory w.r.t. camera
    for demo_name, se3_traj in se3_dict.items():
        depth_dir = './data/'+data_name+'/'+demo_name+'/depth'
        mask_dir = './output/segment/'+data_name+'/'+demo_name
        depth_files = sorted(glob.glob(depth_dir+'/*.npy'))
        mask_files = sorted(glob.glob(mask_dir+'/*.npy'))
        output_path = output_dir+'/'+demo_name
        util.create_dir(output_path, clear = True)

        depth = np.load(depth_files[0])
        mask = np.load(mask_files[0])  
        SE3 = []
        for obj_idx in range(nb_object): 
            com_se3 = vision.get_com(mask, depth, obj_idx)
            network_se3 = se3_traj[0,obj_idx,:]
            # orientation adjusted
            com_se3[3:6] = network_se3[3:6]
            SE3_obj = get_SE3_between_se3s(network_se3, com_se3)
            SE3.append(SE3_obj)
        
        traj = []
        for t in range(len(se3_traj)):
            se3_obj = []
            for obj_idx in range(nb_object):
                SE30 = se3_to_SE3(se3_traj[t,obj_idx,:])
                SE31 = np.matmul(SE3[obj_idx], SE30)

                se31 = SE3_to_se3(SE31)
                se3_obj.append(np.expand_dims(se31,0))
            se3_obj = np.concatenate(se3_obj, 0)
            traj.append(np.expand_dims(se3_obj, 0))
        traj = np.concatenate(traj, 0)

        com_traj = com_dict[demo_name]
        for t in range(len(traj)):
            ax.clear()
            for obj_idx in range(nb_object):
                ax.plot(traj[:t+1,obj_idx,0], traj[:t+1,obj_idx,1], traj[:t+1,obj_idx,2],
                        color[obj_idx], alpha = 0.5, linewidth = 4)
                ax.plot(com_traj[:t+1,obj_idx,0], com_traj[:t+1,obj_idx,1], com_traj[:t+1,obj_idx,2],
                        '--', color = color[obj_idx], alpha = 0.5, linewidth = 4)
                obj.apply_pose(traj[t,obj_idx,:])
                obj.plot(ax, scale = 0.05, linewidth = 3)


                #ax.set_aspect('equal')
            ax.axis('scaled')
            ax.set_xlabel('x(m)')
            ax.set_ylabel('y(m)')
            ax.set_zlabel('z(m)')
            ax.view_init(elev = -60, azim = -90)
            fig.savefig(output_path+'/posecom_%06d.png'%t)

        video_path = output_path+'/object_com_trajectory.avi'
        util.frame_to_video(output_path, video_path, 'png', fps=20)()
    