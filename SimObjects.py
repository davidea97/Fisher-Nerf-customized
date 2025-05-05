import numpy as np
import habitat_sim
import magnum as mn

class SimObject:
    def __init__(self, habitat_obj, name=None, moving=False, show_object_axes=False):
        self.obj = habitat_obj
        self.name = name or f"object_{id(habitat_obj)}"
        self.moving = moving
        self.show_object_axes = show_object_axes

    def get_translation(self):
        return np.round(self.obj.translation, 2)

    def get_rotation_quat(self):
        r = self.obj.rotation
        return np.round(np.array([r.vector.x, r.vector.y, r.vector.z, r.scalar]), 2)
    
    def enable_kinematic_velocity(self, lin_vel, ang_vel, local=True):
        self.obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        vc = self.obj.velocity_control
        vc.linear_velocity = lin_vel
        vc.angular_velocity = ang_vel
        vc.lin_vel_is_local = local
        vc.ang_vel_is_local = local
        vc.controlling_lin_vel = True
        vc.controlling_ang_vel = True
        self.moving = True

    def draw_frame(self, debug_drawer):
        if not self.show_object_axes:
            return
        tf = self.obj.transformation
        length = 0.2
        cone_length = 0.05

        origin = tf.transform_point([0.0, 0.0, 0.0])

        # Axis endpoints (tip of the arrow)
        x_tip = tf.transform_point([length, 0.0, 0.0])
        y_tip = tf.transform_point([0.0, length, 0.0])
        z_tip = tf.transform_point([0.0, 0.0, length])

        # Directions
        x_dir = (x_tip - origin).normalized()
        y_dir = (y_tip - origin).normalized()
        z_dir = (z_tip - origin).normalized()

        # Line shafts
        debug_drawer.draw_transformed_line(origin, x_tip, [1.0, 0.0, 0.0])  # X = red
        debug_drawer.draw_transformed_line(origin, y_tip, [0.0, 1.0, 0.0])  # Y = green
        debug_drawer.draw_transformed_line(origin, z_tip, [0.0, 0.0, 1.0])  # Z = blue

        # Arrowhead cones — now correctly facing *outward*
        debug_drawer.draw_cone(
            translation=x_tip - x_dir * cone_length,  # cone base behind tip
            apex=x_tip,                               # tip of the cone
            radius=0.02,
            color=mn.Color4(1.0, 0.0, 0.0, 1.0),
            normal=x_dir
        )
        debug_drawer.draw_cone(
            translation=y_tip - y_dir * cone_length,
            apex=y_tip,
            radius=0.02,
            color=mn.Color4(0.0, 1.0, 0.0, 1.0),
            normal=y_dir
        )
        debug_drawer.draw_cone(
            translation=z_tip - z_dir * cone_length,
            apex=z_tip,
            radius=0.02,
            color=mn.Color4(0.0, 0.0, 1.0, 1.0),
            normal=z_dir
        )