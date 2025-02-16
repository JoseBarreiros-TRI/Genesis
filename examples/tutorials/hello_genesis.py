import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, -10.0)
        ),
    )

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    # gs.morphs.URDF(
    #     file='urdf/panda_bullet/panda.urdf',
    #     fixed=True,
    # ),
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

scene.build()
for i in range(100):
    scene.step()
