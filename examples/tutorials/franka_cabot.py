import genesis as gs
import numpy as np
import torch
import argparse
import time
import genesis.utils.geom as gu

SIM_DT = 0.01
GYM_DT = 0.1
DEPTH = False

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--cam_vis", action="store_true", default=False)
    parser.add_argument("--n_envs", default=2, type=int)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            plane_reflection=False,
            n_rendered_envs=args.n_envs,
            env_separate_rigid=True,
            show_world_frame=False,
            show_link_frame=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5, 0,2),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=None,  # as fast as possible
        ),
        sim_options=gs.options.SimOptions(
            dt=SIM_DT,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            enable_self_collision=True,
        ),
        # renderer=gs.renderers.RayTracer(  # type: ignore
        #     env_surface=gs.surfaces.Emission(
        #         emissive_texture=gs.textures.ImageTexture(
        #             image_path="textures/indoor_bright.png",
        #         ),
        #     ),
        #     env_radius=15.0,
        #     env_euler=(0, 0, 180),
        #     lights=[
        #         {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
        #     ],
        # ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    # plane = scene.add_entity(
    #     gs.morphs.Plane(),
    # )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.05, 0.04),
            pos=(0.1, 0.0, 0.3),
            euler=(45, 45, 0),
        )
    )
    # banana = scene.add_entity(
    #     gs.morphs.URDF(
    #         file='urdf/fake_banana/fake_banana.urdf',
    #         euler=(0, 0, 0),
    #         pos=(0.15, 0.1, 0.1),
    #         scale=1.0,
    #     )
    # )
    duck = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.1,
            pos=(0.1, -0.3, 0.70),
        ),
    )
    duck2 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.1,
            pos=(0.1, 0.3, 0.70),
            euler=(45, 45, 0),
        ),
    )
    franka_r = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            euler=(0, 0, -45),
            pos=(-0.55, 0.4, -0.129),
            scale=1.0,
        )
        # gs.morphs.URDF(
        #     file='urdf/panda_drake/panda_arm.urdf',
        #     fixed=True,
            # euler=(0, 0, 0),
            # pos=(-0.55, 0.4, -0.129),
            # scale=1.0,
        # ),
    )

    franka_l = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            euler=(0, 0, 45),
            pos=(-0.55, -0.4, -0.129),
            scale=1.0,
        )
        # gs.morphs.URDF(
        #     file='urdf/panda_drake/panda_arm.urdf',
        #     fixed=True,
            # euler=(0, 0, 0),
            # pos=(-0.55, 0.4, -0.129),
            # scale=1.0,
        # ),
    )

    table = scene.add_entity(
        morph=gs.morphs.URDF(
            file='urdf/table/station_table.urdf',
            fixed=True,
            euler=(0, 0, 0),
            pos=(0, 0, 0),
            scale=1.0,
        ),
        # surface=gs.surfaces.Rough(
        #         diffuse_texture=gs.textures.ImageTexture(
        #             image_path="urdf/table/meshes/station_table_top_color.png",
        #         )
        # ),
    )

    cam_ls = []
    cam_rs = []
    cam_wls = []
    cam_wrs = []
    for i in range(1):
        cam_l = scene.add_camera(
            res    = (1280, 720),
            pos    = (1.5, 0.2, 1),
            lookat = (0, 0, 0),
            fov    = 30,
            GUI    = args.cam_vis
        )
        cam_r = scene.add_camera(
            res    = (1280, 720),
            pos    = (1.5, -0.2, 1),
            lookat = (0, 0, 0),
            fov    = 30,
            GUI    = args.cam_vis
        )

        cam_wl = scene.add_camera(
            res    = (1280, 720),
            pos    = (1.5, 0.2, 1),
            lookat = (0, 0, 0),
            fov    = 30,
            GUI    = args.cam_vis
        )
        cam_wr = scene.add_camera(
            res    = (1280, 720),
            pos    = (1.5, -0.2, 1),
            lookat = (0, 0, 0),
            fov    = 30,
            GUI    = args.cam_vis
        )

        cam_ls.append(cam_l)
        cam_rs.append(cam_r)
        cam_wls.append(cam_wl)
        cam_wrs.append(cam_wr)

    cam_w_transform = gu.trans_quat_to_T(np.array([0.03, 0, 0.03]), gu.xyz_to_quat(np.array([180+5, 0, -90])))

    ########################## build ##########################
    n_envs = args.n_envs
    scene.build(n_envs=n_envs, env_spacing=(1.5, 2))

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # set control gains
    franka_l.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka_l.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka_l.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )
    franka_r.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka_r.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka_r.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    ee_l = franka_l.get_link("hand")
    ee_r = franka_r.get_link("hand")

    t0 = time.time()
    franka_l.set_dofs_position([-0.61,-0.61,0,-2.35,0,1.83,0.78], motors_dof)
    franka_r.set_dofs_position([0.436,-0.61,0,-2.35,0,1.83,-1.13], motors_dof)
    # scene.step()
    ee_l = franka_l.get_link("hand")
    ee_r = franka_r.get_link("hand")
    # breakpoint()
    TF = 30 # in sec.
    n_steps = int(TF / GYM_DT)
    for i in range(n_steps):
        if i%(GYM_DT/SIM_DT) ==0:
            SCALE = 0.005
            ee_actions_l = torch.rand((n_envs,3), device="cuda") *SCALE
            ee_actions_r = torch.rand((n_envs,3), device="cuda")*SCALE
            ee_l_xyz = ee_l.get_pos()
            ee_r_xyz = ee_r.get_pos()
            ee_l_quat= ee_l.get_quat()
            ee_r_quat = ee_r.get_quat()
            target_dof_pos_l = ee_actions_l + ee_l_xyz
            target_dof_pos_r = ee_actions_r + ee_r_xyz
            target_dof_quat_l = ee_l_quat
            target_dof_quat_r = ee_r_quat
            qpos_l = franka_l.inverse_kinematics(
                link=ee_l,
                pos=target_dof_pos_l,
                quat=target_dof_quat_l,
            )
            qpos_r = franka_r.inverse_kinematics(
                link=ee_r,
                pos=target_dof_pos_r,
                quat=target_dof_quat_r,
            )
            # breakpoint()
            franka_l.control_dofs_position(qpos_l[:,:-2], motors_dof)
            franka_r.control_dofs_position(qpos_r[:,:-2], motors_dof)

        scene.step()

        if i%(GYM_DT/SIM_DT) ==0:

            Ts_l = gu.trans_quat_to_T(ee_l_xyz, ee_l_quat)
            Ts_r = gu.trans_quat_to_T(ee_r_xyz, ee_r_quat)
            # breakpoint()

            for c_i, cam_wl in enumerate(cam_wls):
                cam_wl.set_pose(transform=Ts_l[c_i].to("cpu").numpy()@ cam_w_transform)
            for c_i, cam_wr in enumerate(cam_wrs):
                cam_wr.set_pose(transform=Ts_r[c_i].to("cpu").numpy()@ cam_w_transform)

            # breakpoint()
            for cam_l in cam_ls:
                color, depth, _, _ = cam_l.render(rgb=True, depth=DEPTH)
            for cam_r in cam_rs:
                color, depth, _, _ = cam_r.render(rgb=True, depth=DEPTH)
            for cam_wl in cam_wls:
                color, depth, _, _ = cam_wl.render(rgb=True, depth=DEPTH)
            for cam_wr in cam_wrs:
                color, depth, _, _ = cam_wr.render(rgb=True, depth=DEPTH)
                # print(color.shape)
        # input("Enter to step..")
    elapsed = time.time()-t0
    print("Elapsed: ",elapsed, " sec", "realtime ratio per env: ", (TF/elapsed)*n_envs)

if __name__ == "__main__":
    main()
