import numpy as np
import habitat_sim
import magnum as mn
import random
class SimObject:
    def __init__(self, habitat_obj, name=None, show_object_axes=False, speed=3.0, dynamic=False):
        self.obj = habitat_obj
        self.name = name or f"object_{id(habitat_obj)}"
        self.show_object_axes = show_object_axes
        self.is_rotating = False
        self.accumulated_rotation = 0.0  
        self.rotation_step = np.pi / 18
        if dynamic:
            self.obj_linear_velocity = np.array([0.0, 0.0, 2.0])
        else:
            self.obj_linear_velocity = np.array([0.0, 0.0, 0.0])
        self.show_object_axes = False
        self.linear_speed = speed

    def get_name(self):
        return self.name
    
    def get_semantic_id(self):
        if hasattr(self.obj, "semantic_id"):
            return self.obj.semantic_id
        else:
            raise AttributeError(f"{self.name} does not have a semantic ID.")
        
    def get_translation(self):
        return np.round(self.obj.translation, 2)
    
    def set_translation(self, translation):
        self.obj.translation = translation

    def set_rotation(self, rotation):
        self.obj.rotation = rotation

    def get_rotation_quat(self):
        r = self.obj.rotation
        return np.round(np.array([r.vector.x, r.vector.y, r.vector.z, r.scalar]), 2)
    
    def get_transformation(self):
        return self.obj.transformation
    
    def get_rotation_step(self):
        return self.rotation_step
    
    def get_accumulated_rotation(self):
        return self.accumulated_rotation
    
    def get_linear_velocity(self):
        return np.round(self.obj_linear_velocity, 2)
    
    def set_rotation_quat(self, quat):
        self.obj.rotation = quat

    def set_rotation_step(self, step):
        self.rotation_step = step

    def enable_kinematic_velocity(self, lin_vel, ang_vel, local=True):
        # self.obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        vc = self.obj.velocity_control
        vc.linear_velocity = lin_vel
        vc.angular_velocity = ang_vel
        vc.lin_vel_is_local = local
        vc.ang_vel_is_local = local
        vc.controlling_lin_vel = True
        vc.controlling_ang_vel = True

    def set_linear_velocity(self, velocity):
        self.obj_linear_velocity = np.array(velocity)

    def get_linear_speed(self):
        return self.linear_speed

    def moving_forward_and_back(self, is_valid):
        
        if self.is_rotating:
            self.enable_kinematic_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            delta_quat = mn.Quaternion.rotation(mn.Rad(self.get_rotation_step()), mn.Vector3.y_axis())
            current_rot = self.obj.rotation
            self.obj.rotation = delta_quat * current_rot
            self.accumulated_rotation += self.get_rotation_step()

            if self.get_accumulated_rotation() >= np.pi:
                self.is_rotating = False
                self.accumulated_rotation = 0.0
        else:
            if not is_valid:
                self.is_rotating = True 
            else:
                self.enable_kinematic_velocity(self.get_linear_velocity().tolist(), [0.0, 0.0, 0.0])
    
    def moving_randomly(self, is_valid):
        if self.is_rotating:
            self.enable_kinematic_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            delta_quat = mn.Quaternion.rotation(mn.Rad(self.get_rotation_step()), mn.Vector3.y_axis())
            current_rot = self.obj.rotation
            self.obj.rotation = delta_quat * current_rot
            self.accumulated_rotation += self.get_rotation_step()

            if self.get_accumulated_rotation() >= np.pi/2:
                self.is_rotating = False
                self.accumulated_rotation = 0.0
                
                theta = random.uniform(-np.pi/4, np.pi/4)  # angle in radians
                direction = np.array([np.cos(theta), 0.0, np.sin(theta)])
                speed = self.get_linear_speed()  # or define a constant
                velocity = direction * speed
                self.set_linear_velocity(velocity)  # <- store this velocity for access in get_linear_velocity()
                yaw = -theta  # because object looks along -Z by default    
                self.set_rotation(mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis()))
        else:
            if not is_valid:
                self.is_rotating = True
            else:
                self.enable_kinematic_velocity(self.get_linear_velocity().tolist(), [0.0, 0.0, 0.0])

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