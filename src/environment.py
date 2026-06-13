# environment.py contains the 2D physics engine and the Reinforcement Learning environment.
# It explicitly provides the AI with pure Proportional Navigation variables
# (LOS Rate, Closing Velocity) and uses a dense distance-closing reward
# to encourage natural intercept learning without imitation.
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Plane:
    def __init__(self, x, y, speed, heading, mass, thrust, drag, dt):
        self.x = x
        self.y = y
        self.speed = speed
        self.heading = heading
        self.mass = mass
        self.thrust = thrust
        self.drag = drag
        self.dt = dt

        self.vel_x = speed*np.cos(self.heading)
        self.vel_y = speed*np.sin(self.heading)

    def update(self, rotate):
        #Thrust Components
        force_x = self.thrust * np.cos(self.heading)
        force_y = self.thrust * np.sin(self.heading)

        #Drag Force
        force_x -= self.drag * (self.speed**2) * np.cos(self.heading)
        force_y -= self.drag * (self.speed**2) * np.sin(self.heading)

        #Acceleration Components
        acc_x = force_x / self.mass
        acc_y = force_y / self.mass

        #Velocity Components
        self.vel_x += acc_x * self.dt
        self.vel_y += acc_y * self.dt

        self.speed = np.sqrt(self.vel_x**2 + self.vel_y**2)
        
        # Aerodynamic stability: align velocity with heading
        self.vel_x = self.speed * np.cos(self.heading)
        self.vel_y = self.speed * np.sin(self.heading)

        #Position Components
        self.x += self.vel_x * self.dt
        self.y += self.vel_y * self.dt

        #Rotation
        self.heading += rotate * self.dt

        #Rotation Normalize
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi
    
class Missile:
    def __init__(self, x, y, speed, heading, mass, thrust, drag, max_rotate, dt):
        self.x = x
        self.y = y
        self.speed = speed
        self.heading = heading
        self.mass = mass
        self.thrust = thrust
        self.drag = drag
        self.dt = dt
        self.max_rotate = max_rotate

        self.vel_x = speed*np.cos(self.heading)
        self.vel_y = speed*np.sin(self.heading)

    def update(self, action):

        #Turn Command
        turn_command = action[0] * self.max_rotate
        self.heading += turn_command * self.dt
        
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        #Thrust Components
        force_x = self.thrust * np.cos(self.heading)
        force_y = self.thrust * np.sin(self.heading)

        #Drag Force
        force_x -= self.drag * (self.speed**2) * np.cos(self.heading)
        force_y -= self.drag * (self.speed**2) * np.sin(self.heading)

        #Acceleration Components
        acc_x = force_x / self.mass
        acc_y = force_y / self.mass

        #Velocity Components
        self.vel_x += acc_x * self.dt
        self.vel_y += acc_y * self.dt

        self.speed = np.sqrt(self.vel_x**2 + self.vel_y**2)

        # 3. AERODYNAMIC STABILITY (The Anti-Drift Fix)
        # We force the velocity vector to point in the same direction as the nose
        # of the missile. This simulates fins preventing sideways slipping.
        self.vel_x = self.speed * np.cos(self.heading)
        self.vel_y = self.speed * np.sin(self.heading)

        #Position Components
        self.x += self.vel_x * self.dt
        self.y += self.vel_y * self.dt

class MissileEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Handles the 2D physics of a missile chasing a plane.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, truncation_mode="advanced"):
        super().__init__()
        self.render_mode = render_mode
        self.dt = 1.0 / 60.0  # Simulation time step
        # Simulation time (seconds) - advances each step
        self.sim_time = 0.0
        
        # Truncation Mode: "simple" only uses distance boundaries.
        # "advanced" adds a start-distance runaway cutoff as a learning heuristic.
        self.truncation_mode = truncation_mode  # "simple" or "advanced"

        # 1. ACTION SPACE (What the AI can do)
        # Example: Continuous control of turning rate (angular velocity)
        # Values between -1.0 (max left) and 1.0 (max right)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 2. OBSERVATION SPACE (What the AI can see)
        # We explicitly provide the processed kinematic variables necessary for Proportional Navigation:
        # [los_rate_norm, closing_velocity_norm]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        # Plane control: can be set via `set_plane_control` to a number, callable(t)->rotate, or string expression using `t`.
        self.plane_control_callable = None
        self._plane_control_code = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset simulation time
        self.sim_time = 0.0
        
        if self.render_mode is None:
            # ===== TRAINING MODE: Smart Random Environment =====
            # Physics parameters for different plane/missile combinations
            # are randomly selected. Each sample is tested with a simple
            # proportional navigation algorithm; if it hits, the setup is accepted.

            missile_x = 0.0
            missile_y = 0.0

            max_attempts = 6
            chosen_params = None
            feasible = False

            for _ in range(max_attempts):
                # Geometry: Random spawning distance
                distance = np.random.uniform(1000.0, 3500.0)
                angle = np.random.uniform(-np.pi, np.pi)
                plane_x = missile_x + distance * np.cos(angle)
                plane_y = missile_y + distance * np.sin(angle)

                # Target angle relative to missile
                target_angle = np.arctan2(plane_y - missile_y, plane_x - missile_x)
                missile_heading = target_angle + np.random.uniform(-np.pi/2, np.pi/2)
                missile_heading = (missile_heading + np.pi) % (2 * np.pi) - np.pi
                
                # Plane heading fully random
                plane_heading = np.random.uniform(-np.pi, np.pi)

                # Random physics parameters (widened bounds for robust generalization)
                missile_speed = float(np.random.uniform(350.0, 550.0))
                missile_mass = float(np.random.uniform(50.0, 120.0))
                missile_thrust = float(np.random.uniform(400.0, 1500.0))
                missile_drag = float(np.random.uniform(0.002, 0.008))
                missile_max_rotate = float(np.random.uniform(0.3, 1.2))

                plane_speed = float(np.random.uniform(80.0, 200.0))
                plane_mass = float(np.random.uniform(350.0, 650.0))
                plane_thrust = float(np.random.uniform(100.0, 400.0))
                plane_drag = float(np.random.uniform(0.010, 0.020))

                params = {
                    "missile": {"speed": missile_speed, "mass": missile_mass, "thrust": missile_thrust, "drag": missile_drag, "max_rotate": missile_max_rotate},
                    "plane": {"speed": plane_speed, "mass": plane_mass, "thrust": plane_thrust, "drag": plane_drag},
                    "geometry": {"plane_x": plane_x, "plane_y": plane_y, "missile_x": missile_x, "missile_y": missile_y, "missile_heading": missile_heading, "plane_heading": plane_heading}
                }

                if self._feasibility_test(params, max_time=200.0):
                    chosen_params = params
                    feasible = True
                    break

            # If none feasible found, keep the last sampled params but mark infeasible
            if chosen_params is None:
                chosen_params = params
                feasible = False

            # Unpack chosen params for instantiation below
            missile_speed = chosen_params["missile"]["speed"]
            missile_mass = chosen_params["missile"]["mass"]
            missile_thrust = chosen_params["missile"]["thrust"]
            missile_drag = chosen_params["missile"]["drag"]
            missile_max_rotate = chosen_params["missile"]["max_rotate"]

            plane_speed = chosen_params["plane"]["speed"]
            plane_mass = chosen_params["plane"]["mass"]
            plane_thrust = chosen_params["plane"]["thrust"]
            plane_drag = chosen_params["plane"]["drag"]

            plane_x = chosen_params["geometry"]["plane_x"]
            plane_y = chosen_params["geometry"]["plane_y"]
            missile_heading = chosen_params["geometry"]["missile_heading"]
            plane_heading = chosen_params["geometry"]["plane_heading"]
            
            # Apply Random Training Behaviors
            behavior_type = np.random.choice([1, 2, 3])
            if behavior_type == 1:
                # 1. Straight
                self.set_plane_control(0.0)
            elif behavior_type == 2:
                # 2. Loiter (Tight Circle)
                radius = np.random.uniform(200.0, 800.0)
                turn_rate = plane_speed / radius
                if np.random.rand() > 0.5:
                    turn_rate = -turn_rate
                self.set_plane_control(turn_rate)
            elif behavior_type == 3:
                # 3. Harmonic Wave (High G-force weaving)
                a = np.random.uniform(0.4, 1.2)
                b = np.random.uniform(0.5, 2.0)
                self.set_plane_control(f"{a:.3f}*sin({b:.3f}*t)")
                
            self.feasible = feasible
        else:
            # TEST MODE (render_mode="human"): Realistic starting distance
            # Closer starting distances during testing align better with training results.
            distance = np.random.uniform(1800.0, 3200.0)
            angle = np.random.uniform(-np.pi, np.pi)
            plane_x = distance * np.cos(angle)
            plane_y = distance * np.sin(angle)
            
            missile_x = 0.0
            missile_y = 0.0
            
            # Missile must face the plane initially in test mode
            target_angle = np.arctan2(plane_y - missile_y, plane_x - missile_x)
            missile_heading = target_angle + np.random.uniform(-np.pi/6, np.pi/6)
            missile_heading = (missile_heading + np.pi) % (2 * np.pi) - np.pi
            
            plane_heading = np.random.uniform(-np.pi, np.pi)
            
            # Aggressive missile parameters in test mode
            missile_speed = 450.0
            missile_mass = 70.0
            missile_thrust = 800.0
            missile_drag = 0.003
            missile_max_rotate = 0.5
            
            plane_speed = 250.0
            plane_mass = 450.0
            plane_thrust = 150.0
            plane_drag = 0.015
            self.feasible = False
            chosen_params = {
                "missile": {"speed": missile_speed, "mass": missile_mass, "thrust": missile_thrust, "drag": missile_drag, "max_rotate": missile_max_rotate},
                "plane": {"speed": plane_speed, "mass": plane_mass, "thrust": plane_thrust, "drag": plane_drag},
                "geometry": {"plane_x": plane_x, "plane_y": plane_y, "missile_x": missile_x, "missile_y": missile_y, "missile_heading": missile_heading, "plane_heading": plane_heading},
            }
        
        # Store starting positions for miss detection (advanced mode)
        self.missile_start_x = missile_x
        self.missile_start_y = missile_y
        
        # Re-instantiate the classes (using your dynamic physics variables)
        self.plane = Plane(plane_x, plane_y, speed=plane_speed, heading=plane_heading, mass=plane_mass, thrust=plane_thrust, drag=plane_drag, dt=self.dt)
        self.missile = Missile(missile_x, missile_y, speed=missile_speed, heading=missile_heading, mass=missile_mass, thrust=missile_thrust, drag=missile_drag, max_rotate=missile_max_rotate, dt=self.dt)
        
        # Initialize the distance tracker for the reward function
        dist_x = self.plane.x - self.missile.x
        dist_y = self.plane.y - self.missile.y
        self.previous_distance = np.sqrt(dist_x**2 + dist_y**2)
        self.initial_plane_missile_distance = self.previous_distance  # Store for miss detection
        
        observation = self._get_obs()
        info = {}
        
        # If user passed a plane_control in options, set it now (supports numeric, callable, or string)
        if options and isinstance(options, dict) and "plane_control" in options:
            self.set_plane_control(options["plane_control"])

        info["feasible"] = getattr(self, "feasible", False)
        info["params"] = chosen_params

        return observation, info

    def set_plane_control(self, control):
        if control is None:
            self.plane_control_callable = None
            self._plane_control_code = None
            return

        if callable(control):
            self.plane_control_callable = control
            self._plane_control_code = None
            return

        if isinstance(control, (int, float, np.integer, np.floating)):
            val = float(control)
            self.plane_control_callable = lambda t, v=val: v
            self._plane_control_code = None
            return

        if isinstance(control, str):
            try:
                code = compile(control, '<plane_control>', 'eval')
                self._plane_control_code = code
                self.plane_control_callable = None
                return
            except Exception as e:
                raise ValueError(f"Invalid plane_control expression: {e}")

        raise TypeError("plane_control must be None, number, callable, or string expression")

    def _feasibility_test(self, params, max_time=200.0):
        try:
            dt = self.dt  # Use exact simulation dt to ensure perfectly accurate feasibility checks
            missile_p = params["missile"]
            plane_p = params["plane"]
            geo = params["geometry"]

            missile = Missile(geo["missile_x"], geo["missile_y"], speed=missile_p["speed"], heading=geo["missile_heading"], mass=missile_p["mass"], thrust=missile_p["thrust"], drag=missile_p["drag"], max_rotate=missile_p["max_rotate"], dt=dt)
            plane = Plane(geo["plane_x"], geo["plane_y"], speed=plane_p["speed"], heading=geo["plane_heading"], mass=plane_p["mass"], thrust=plane_p["thrust"], drag=plane_p["drag"], dt=dt)

            initial_distance = np.hypot(plane.x - missile.x, plane.y - missile.y)
            missile_start_x = missile.x
            missile_start_y = missile.y

            steps = int(max_time / dt)
            for _ in range(steps):
                dx = plane.x - missile.x
                dy = plane.y - missile.y
                distance = np.hypot(dx, dy)
                target_angle = np.arctan2(dy, dx)
                
                # Proportional Navigation
                rel_vx = plane.vel_x - missile.vel_x
                rel_vy = plane.vel_y - missile.vel_y
                los_rate = (dx * rel_vy - dy * rel_vx) / (distance**2 + 1e-6)
                
                N = 4.0
                vc = -(rel_vx * dx + rel_vy * dy) / (distance + 1e-6)
                if vc < 0:
                    vc = 1.0 
                    
                desired_turn_rate = (N * vc * los_rate) / (missile.speed + 1e-6)
                
                # Blend with Pure Pursuit if heading error is large
                angle_error = (target_angle - missile.heading + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_error) > np.radians(20):
                    desired_turn_rate += angle_error * 1.5

                desired_turn_rate = np.clip(desired_turn_rate, -missile.max_rotate, missile.max_rotate)
                action = np.array([desired_turn_rate / missile.max_rotate], dtype=np.float32)

                missile.update(action)
                plane.update(rotate=0.0)

                distance = np.hypot(plane.x - missile.x, plane.y - missile.y)
                if distance < 40.0:
                    return True

                # OPTIMIZATION: Feasibility step uses expanded 3000 boundary
                if distance > max(4500.0, initial_distance * 1.3):
                    return False
                    
                # Dynamic Tail Chase Failure Condition
                # If they are going about the same heading, and the missile is no longer significantly faster
                heading_error = (plane.heading - missile.heading + np.pi) % (2 * np.pi) - np.pi
                if abs(heading_error) < np.radians(15):
                    if missile.speed - plane.speed < 5.0:
                        accel = (missile.thrust - missile.drag * (missile.speed**2)) / missile.mass
                        future_speed = missile.speed + accel * 4.0
                        if future_speed - plane.speed < 5.0:
                            return False

            return False
        except Exception:
            return False

    def step(self, action):
        # 1. Normalize and validate action from policy
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            action = action.ravel()[: self.action_space.shape[0]]
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 2. Update Physics
        self.missile.update(action)

        # Determine plane rotate command from configured controller
        rotate = 0.0
        try:
            if self.plane_control_callable is not None:
                rotate = float(self.plane_control_callable(self.sim_time))
            elif self._plane_control_code is not None:
                safe_globals = {
                    "__builtins__": None,
                    "np": np,
                    "sin": np.sin,
                    "cos": np.cos,
                    "tan": np.tan,
                    "pi": np.pi,
                    "t": self.sim_time,
                }
                rotate = float(eval(self._plane_control_code, safe_globals, {}))
        except Exception:
            rotate = 0.0

        # Apply rotation to plane and advance sim time
        self.plane.update(rotate=rotate)
        self.sim_time += self.dt

        # 2. Calculate Distance
        dist_x = self.plane.x - self.missile.x
        dist_y = self.plane.y - self.missile.y
        distance = np.sqrt(dist_x**2 + dist_y**2)

        # 3. Check Termination / Truncation
        terminated = bool(distance < 40.0)      # Hit Radius
        
        # Miss Detection: Calculate if missile has flown past the plane
        missile_dist_from_start = np.hypot(self.missile.x - self.missile_start_x, self.missile.y - self.missile_start_y)
        plane_dist_from_missile_start = np.hypot(self.plane.x - self.missile_start_x, self.plane.y - self.missile_start_y)
        miss_condition = missile_dist_from_start > (1.2 * plane_dist_from_missile_start)
        
        # OPTIMIZATION: Dynamic max distance limit to allow flexible boundaries
        max_distance_limit = max(4800.0, self.initial_plane_missile_distance * 1.4)

        if self.truncation_mode == "advanced":
            boundary_condition = distance > max_distance_limit
            
            # Dynamic Tail Chase Failure Condition
            heading_error = (self.plane.heading - self.missile.heading + np.pi) % (2 * np.pi) - np.pi
            
            accel = (self.missile.thrust - self.missile.drag * (self.missile.speed**2)) / self.missile.mass
            future_speed = self.missile.speed + accel * 4.0
            
            tail_chase_failure = abs(heading_error) < np.radians(15) and (self.missile.speed - self.plane.speed < 5.0) and (future_speed - self.plane.speed < 5.0)
            
            truncated = bool(miss_condition or boundary_condition or tail_chase_failure)
        else:
            heading_error = (self.plane.heading - self.missile.heading + np.pi) % (2 * np.pi) - np.pi
            
            accel = (self.missile.thrust - self.missile.drag * (self.missile.speed**2)) / self.missile.mass
            future_speed = self.missile.speed + accel * 4.0
            
            tail_chase_failure = abs(heading_error) < np.radians(15) and (self.missile.speed - self.plane.speed < 5.0) and (future_speed - self.plane.speed < 5.0)
            
            truncated = bool(distance > max_distance_limit or miss_condition or tail_chase_failure)

        # 4. Centralized Reward Call
        reward = self._compute_reward(distance, terminated, truncated, action)

        # 5. State Update
        self.previous_distance = distance
            
        # 6. Get Observation
        observation = self._get_obs()
        
        return observation, reward, terminated, truncated, {}

    def render(self):
        pass

    def _get_obs(self):
        dx = self.plane.x - self.missile.x
        dy = self.plane.y - self.missile.y
        theta = self.missile.heading
        
        local_x = dx * np.cos(theta) + dy * np.sin(theta)
        local_y = -dx * np.sin(theta) + dy * np.cos(theta)
        
        dvx = self.plane.vel_x - self.missile.vel_x
        dvy = self.plane.vel_y - self.missile.vel_y
        
        distance = np.hypot(dx, dy)
        
        distance_sq = distance**2
        los_rate = (dx * dvy - dy * dvx) / (distance_sq + 1e-6)
        vc = -(dx * dvx + dy * dvy) / (distance + 1e-6)

        # TRUE MISSILE OBSERVATIONS (Processed variables for Proportional Navigation)
        # 1. Line of Sight Rate (LOS Rate)
        los_rate_norm = np.clip(los_rate, -1.0, 1.0)
        # 2. Closing Velocity (Vc)
        vc_norm = np.clip(vc / 500.0, -1.0, 1.0)

        obs = np.array([
            los_rate_norm,       
            vc_norm
        ], dtype=np.float32)
        
        return obs

    def _compute_reward(self, distance, terminated, truncated, action=None):
        if terminated:
            return 1000.0  # Massive Success reward
        if truncated:
            return -500.0  # Massive Boundary/Timeout penalty

        if action is None:
            return 0.0

        # Proper Training Evaluation
        # Dense reward for getting closer to the target
        progress = self.previous_distance - distance
        getting_close_reward = progress * 0.1
        
        # Small time penalty to encourage fast interceptions
        time_penalty = -0.1
        
        return getting_close_reward + time_penalty