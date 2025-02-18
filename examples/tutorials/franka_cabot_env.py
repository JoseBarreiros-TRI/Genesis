import genesis as gs
import numpy as np
import torch
import argparse
import time


class FrankaCabotEnv:
    def __init__(self, num_envs, env_cfg, reward_cfg, show_viewer=False, show_cam_viewer=False, device="cuda"):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.gym_dt = 0.1  # control frequency on real robot is 10hz
        self.sim_dt = 0.01
        self.max_episode_length = int(env_cfg["episode_length_s"] / self.gym_dt)
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg

        ###### create a scene ####
        self.scene = gs.Scene(
            vis_options=gs.options.VisOptions(
                plane_reflection=False,
                n_rendered_envs=self.num_envs,
                env_separate_rigid=True,
                show_world_frame=False,
                show_link_frame=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=None,  # as fast as possible
            ),
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_joint_limit=True,
                enable_collision=True,
                enable_self_collision=True,
            ),
            show_viewer=show_viewer,
        )

        ######## entities #############
        self.franka_r = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                euler=(0, 0, -45),
                pos=(-0.55, 0.4, -0.129),
                scale=1.0,
            )
        )

        self.franka_l = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                euler=(0, 0, 45),
                pos=(-0.55, -0.4, -0.129),
                scale=1.0,
            )
        )

        table = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file='urdf/table/station_table.urdf',
                fixed=True,
                euler=(0, 0, 0),
                pos=(0, 0, 0),
                scale=1.0,
            ),
        )

        cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.05, 0.04),
                pos=(0.1, 0.0, 0.05),
            )
        )

        cam_l = self.scene.add_camera(
            res    = (1280, 720),
            pos    = (1.5, 0.2, 1),
            lookat = (0, 0, 0),
            fov    = 30,
            GUI    = show_cam_viewer
        )
        cam_r = self.scene.add_camera(
            res    = (1280, 720),
            pos    = (1.5, -0.2, 1),
            lookat = (0, 0, 0),
            fov    = 30,
            GUI    = show_cam_viewer
        )

        cam_wl = self.scene.add_camera(
            res    = (1280, 720),
            pos    = (1.5, 0.2, 1),
            lookat = (0, 0, 0),
            fov    = 30,
            GUI    = show_cam_viewer
        )
        cam_wr = self.scene.add_camera(
            res    = (1280, 720),
            pos    = (1.5, -0.2, 1),
            lookat = (0, 0, 0),
            fov    = 30,
            GUI    = show_cam_viewer
        )
        self.cameras = {
            "scene_left": cam_l,
            "scene_right": cam_r,
            "wrist_left": cam_wl,
            "wrist_right": cam_wr,
        }
        self.scene.build(n_envs=num_envs, env_spacing=(1.5, 2))
        # local indices
        self.motors_dofs = np.arange(7)
        self.fingers_dofs = np.arange(7, 9)

        # set control gains
        self.franka_l.set_dofs_kp(env_cfg["franka_kps"])
        self.franka_r.set_dofs_kp(env_cfg["franka_kps"])
        self.franka_l.set_dofs_kv(env_cfg["franka_kvs"])
        self.franka_r.set_dofs_kv(env_cfg["franka_kvs"])
        # breakpoint()
        self.franka_l.set_dofs_force_range(
            -1 * np.array(env_cfg["franka_force_lim"]),
            np.array(env_cfg["franka_force_lim"]),
        )
        self.franka_r.set_dofs_force_range(
            -1 * np.array(env_cfg["franka_force_lim"]),
            np.array(env_cfg["franka_force_lim"]),
        )

        self.ee_l = self.franka_l.get_link("hand")
        self.ee_r = self.franka_r.get_link("hand")

        self.obs_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.extras = dict()  # extra information for logging

    def step(self, actions):
        # actions:
        # xyz_l, rot6d_l, xyz_r, rot6d_r, gripper_l, gripper_r
        actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        actions *= self.env_cfg["action_scale"]

        ee_xyz_l = actions[:, 0:3]
        ee_rot6d_l = actions[:, 3:9]
        ee_xyz_r = actions[:, 9:12]
        ee_rot6d_r = actions[:, 12:18]
        gripper_l = actions[:, 18:19]
        gripper_r = actions[:, 19:20]

        target_dof_pos_l = ee_xyz_l + self.ee_l.get_pos()
        target_dof_pos_r = ee_xyz_r + self.ee_r.get_pos()
        target_dof_quat_l = self.ee_l.get_quat()
        target_dof_quat_r = self.ee_r.get_quat()
        target_gripper_l = gripper_l + 0
        target_gripper_r = gripper_r + 0

        qpos_l = self.franka_l.inverse_kinematics(
            link=self.ee_l,
            pos=target_dof_pos_l,
            quat=target_dof_quat_l,
        )
        qpos_r = self.franka_r.inverse_kinematics(
            link=self.ee_r,
            pos=target_dof_pos_r,
            quat=target_dof_quat_r,
        )
        self.franka_l.control_dofs_position(qpos_l[:,:-2], self.motors_dofs)
        self.franka_r.control_dofs_position(qpos_r[:,:-2], self.motors_dofs)
        # self.franka_l.control_dofs_position(target_gripper_l, self.fingers_dofs)
        # self.franka_r.control_dofs_position(target_gripper_r, self.fingers_dofs)

        self.scene.step()

        # move wrist cameras
        self.cameras["wrist_left"].set_pose(transform=0)
        self.cameras["wrist_right"].set_pose(transform=0)

        for cam in self.cameras.items():
            rgb, depth, segmentation, normal = cam.render()

        self.obs_buf = torch.cat(
            [
            ],
            axis=-1,
        )
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf


    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.franka_l.set_dofs_position(
            self.env_cfg["franka_l_q0"], self.motors_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
            )
        self.franka_r.set_dofs_position(
            self.env_cfg["franka_r_q0"], self.motors_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
            )

        # fill extras
        self.extras["episode"] = {}

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--num_envs", default=2, type=int)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    env_cfg = {
        "episode_length_s": 10,
        "franka_kps": [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
        "franka_kvs": [450, 450, 350, 350, 200, 200, 200, 10, 10],
        "franka_force_lim": [87, 87, 87, 87, 12, 12, 12, 100, 100],
        "clip_actions": 1,
        "action_scale": 0.005,
        "franka_l_q0": [-0.61,-0.61,0,-2.35,0,1.83,0.78],
        "franka_r_q0": [0.436,-0.61,0,-2.35,0,1.83,-1.13],
    }
    reward_cfg = {}

    env = FrankaCabotEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
        show_cam_viewer=args.vis,
        device=args.device,
    )
    t0 = time.time()
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = torch.rand((args.num_envs, 20), device=args.device)
            obs, _, rews, dones, infos = env.step(actions)

    elapsed = time.time()-t0
    # print("Elapsed: ",elapsed, " sec", "realtime ratio per env: ", (TF/elapsed)*n_envs)

if __name__ == "__main__":
    main()
