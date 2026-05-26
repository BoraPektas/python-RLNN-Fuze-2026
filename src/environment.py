#environment.py fizik motoru ve geri kalan çevreyi içermektedir.
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

    def __init__(self, render_mode=None, truncation_mode="simple"):
        super().__init__()
        self.render_mode = render_mode
        self.dt = 1.0 / 60.0  # Simulation time step
        # Simulation time (seconds) - advances each step
        self.sim_time = 0.0
        
        # Truncation Mode: "simple" only uses distance > 3000.0.
        # "advanced" adds a start-distance runaway cutoff as a learning heuristic.
        self.truncation_mode = truncation_mode  # "simple" or "advanced"

        # 1. ACTION SPACE (What the AI can do)
        # Example: Continuous control of turning rate (angular velocity)
        # Values between -1.0 (max left) and 1.0 (max right)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 2. OBSERVATION SPACE (What the AI can see)
        # Example: [relative x, relative y, relative velocity x, relative velocity y, heading cos, heading sin, speed delta]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        # Plane control: can be set via `set_plane_control` to a number, callable(t)->rotate, or string expression using `t`.
        self.plane_control_callable = None
        self._plane_control_code = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset simulation time
        self.sim_time = 0.0
        
        if self.render_mode is None:
            # ===== EĞİTİM MODU: Mantıklı Rastgele Ortam =====
            # Farklı uçak/füze kombinasyonları için fiziksel parametreler
            # rastgele seçilir. Her örnek, basit bir oransal-seyrüsefer
            # (proportional guidance) ile test edilir; vuruyorsa kabul edilir.

            missile_x = 0.0
            missile_y = 0.0

            max_attempts = 6
            chosen_params = None
            feasible = False

            for _ in range(max_attempts):
                # Geometry: plane at 1800-3500 units around missile (easier for training)
                distance = np.random.uniform(1800.0, 3500.0)
                angle = np.random.uniform(-np.pi, np.pi)
                plane_x = missile_x + distance * np.cos(angle)
                plane_y = missile_y + distance * np.sin(angle)

                # Initial headings with some spread
                target_angle = np.arctan2(plane_y - missile_y, plane_x - missile_x)
                missile_heading = target_angle + np.random.uniform(-np.pi/4, np.pi/4)
                missile_heading = (missile_heading + np.pi) % (2 * np.pi) - np.pi
                plane_heading = missile_heading + np.random.uniform(-np.pi/2, np.pi/2)
                plane_heading = (plane_heading + np.pi) % (2 * np.pi) - np.pi

                # Random physics parameters (optimized for better learning)
                missile_speed = float(np.random.uniform(350.0, 480.0))
                missile_mass = float(np.random.uniform(50.0, 120.0))
                missile_thrust = float(np.random.uniform(600.0, 900.0))
                missile_drag = float(np.random.uniform(0.002, 0.008))
                missile_max_rotate = float(np.random.uniform(2.5, 4.5))

                plane_speed = float(np.random.uniform(80.0, 140.0))
                plane_mass = float(np.random.uniform(350.0, 650.0))
                plane_thrust = float(np.random.uniform(100.0, 250.0))
                plane_drag = float(np.random.uniform(0.010, 0.020))

                params = {
                    "missile": {"speed": missile_speed, "mass": missile_mass, "thrust": missile_thrust, "drag": missile_drag, "max_rotate": missile_max_rotate},
                    "plane": {"speed": plane_speed, "mass": plane_mass, "thrust": plane_thrust, "drag": plane_drag},
                    "geometry": {"plane_x": plane_x, "plane_y": plane_y, "missile_x": missile_x, "missile_y": missile_y, "missile_heading": missile_heading, "plane_heading": plane_heading}
                }

                if self._feasibility_test(params, max_time=20.0):
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
            self.feasible = feasible
            
        else:
            # ===== TEST MODU (render_mode="human"): Yine zorlu ama daha gerçekçi bir başlangıç mesafesi
            # Uçakların füzeye daha yakın olduğu testler, eğitimle daha uyumlu sonuç verir.
            distance = np.random.uniform(1800.0, 3200.0)
            angle = np.random.uniform(-np.pi, np.pi)
            plane_x = distance * np.cos(angle)
            plane_y = distance * np.sin(angle)
            plane_heading = np.random.uniform(-np.pi, np.pi)
            
            missile_x = 0.0
            missile_y = 0.0
            
            # Füze uçağa doğru bakmalı (test modunda da)
            target_angle = np.arctan2(plane_y - missile_y, plane_x - missile_x)
            missile_heading = target_angle + np.random.uniform(-np.pi/6, np.pi/6)
            missile_heading = (missile_heading + np.pi) % (2 * np.pi) - np.pi
            
            # Test modunda agresif füze parametreleri
            missile_speed = 450.0
            missile_mass = 70.0
            missile_thrust = 800.0
            missile_drag = 0.003
            missile_max_rotate = 4.0
            
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
        """
        Set the plane control. `control` can be:
        - None: disable control (defaults used)
        - number: constant rotation rate (radians/sec)
        - callable: function f(t)->rotate (radians/sec)
        - string: expression evaluated with `t` in seconds and `np` available, e.g. "0.2", "np.sin(t)", "0.5*np.sin(t)"
        """
        if control is None:
            self.plane_control_callable = None
            self._plane_control_code = None
            return

        if callable(control):
            self.plane_control_callable = control
            self._plane_control_code = None
            return

        # Numeric literal
        if isinstance(control, (int, float, np.integer, np.floating)):
            val = float(control)
            self.plane_control_callable = lambda t, v=val: v
            self._plane_control_code = None
            return

        # String expression: compile once
        if isinstance(control, str):
            try:
                code = compile(control, '<plane_control>', 'eval')
                self._plane_control_code = code
                self.plane_control_callable = None
                return
            except Exception as e:
                raise ValueError(f"Invalid plane_control expression: {e}")

        raise TypeError("plane_control must be None, number, callable, or string expression")

    def _feasibility_test(self, params, max_time=20.0):
        """
        Run a short simulation using a simple proportional guidance controller
        to check whether the randomly-generated missile can intercept the plane.
        Returns True if hit occurs within max_time, else False.
        """
        try:
            dt = self.dt
            missile_p = params["missile"]
            plane_p = params["plane"]
            geo = params["geometry"]

            missile = Missile(geo["missile_x"], geo["missile_y"], speed=missile_p["speed"], heading=geo["missile_heading"], mass=missile_p["mass"], thrust=missile_p["thrust"], drag=missile_p["drag"], max_rotate=missile_p["max_rotate"], dt=dt)
            plane = Plane(geo["plane_x"], geo["plane_y"], speed=plane_p["speed"], heading=geo["plane_heading"], mass=plane_p["mass"], thrust=plane_p["thrust"], drag=plane_p["drag"], dt=dt)

            dist_x = plane.x - missile.x
            dist_y = plane.y - missile.y
            previous_distance = np.sqrt(dist_x**2 + dist_y**2)
            initial_plane_missile_distance = previous_distance
            missile_start_x = missile.x
            missile_start_y = missile.y

            steps = int(max_time / dt)
            for _ in range(steps):
                # Simple proportional heading controller: aim towards target
                dx = plane.x - missile.x
                dy = plane.y - missile.y
                target_angle = np.arctan2(dy, dx)
                angle_error = (target_angle - missile.heading + np.pi) % (2 * np.pi) - np.pi

                # Gain chosen empirically for feasibility check
                K = 2.0
                desired_turn_rate = angle_error * K
                # Clip to missile's physical capability
                desired_turn_rate = np.clip(desired_turn_rate, -missile.max_rotate, missile.max_rotate)
                action = np.array([desired_turn_rate / missile.max_rotate], dtype=np.float32)

                missile.update(action)
                # For feasibility check use the same default steering profile as the main env
                plane.update(rotate=0.0)

                distance = np.hypot(plane.x - missile.x, plane.y - missile.y)
                if distance < 20.0:
                    return True

                missile_dist_from_start = np.hypot(missile.x - missile_start_x, missile.y - missile_start_y)
                if missile_dist_from_start > initial_plane_missile_distance:
                    return False
                if distance > 3000.0:
                    return False

            return False
        except Exception:
            # On any error, mark as infeasible (conservative approach)
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
                # Safe eval environment: expose numpy and common trig helpers, and the current time `t`
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
            # On any controller error, fall back to no automatic rotation
            rotate = 0.0

        # Apply rotation to plane and advance sim time
        self.plane.update(rotate=rotate)
        self.sim_time += self.dt

        # 2. Calculate Distance
        dist_x = self.plane.x - self.missile.x
        dist_y = self.plane.y - self.missile.y
        distance = np.sqrt(dist_x**2 + dist_y**2)

        # 3. Check Termination / Truncation
        terminated = bool(distance < 20.0)      # Hit Radius
        
        # Miss Detection: Calculate missile distance from starting point
        missile_dist_from_start_x = self.missile.x - self.missile_start_x
        missile_dist_from_start_y = self.missile.y - self.missile_start_y
        missile_dist_from_start = np.sqrt(missile_dist_from_start_x**2 + missile_dist_from_start_y**2)
        
        # Truncation logic based on mode
        if self.truncation_mode == "advanced":
            # ADVANCED: Allow daha uzun manevra süresi.
            # Paths are çoğu zaman dümdüz olmayan bir rota çizdiği için,
            # missile path length'in başlangıç mesafesinin 2 katından fazla olması
            # hâlinde artık gerçekçi bir kaçış olduğunu kabul ediyoruz.
            miss_condition = missile_dist_from_start > 2.0 * self.initial_plane_missile_distance
            boundary_condition = distance > 3000.0
            truncated = bool(miss_condition or boundary_condition)
        else:  # "simple" mode
            # SIMPLE: Original truncation - just max distance
            truncated = bool(distance > 3000.0)

        # 4. Centralized Reward Call
        reward = self._compute_reward(distance, terminated, truncated)

        # 5. State Update
        self.previous_distance = distance
            
        # 6. Get Observation
        observation = self._get_obs()
        
        return observation, reward, terminated, truncated, {}

    def render(self):
        """
        (Optional but recommended) Hooks into Pygame to draw the current 
        state if render_mode == "human". 
        Since you have a separate Gui.py, you might handle rendering there, 
        but Gym standard practice is to allow rendering from the env directly.
        """
        pass

    def _get_obs(self):
        """
        Calculates the state of the environment relative to the missile's body frame.
        """
        # 1. Calculate global positional differences
        dx = self.plane.x - self.missile.x
        dy = self.plane.y - self.missile.y
        
        # 2. Get the missile's current heading
        theta = self.missile.heading
        
        # 3. Apply Trigonometric Rotation Matrix (Local Body Frame)
        local_x = dx * np.cos(theta) + dy * np.sin(theta)
        local_y = -dx * np.sin(theta) + dy * np.cos(theta)
        
        # 5. Calculate Relative Velocity (Optional but highly recommended)
        # It helps the AI lead the target instead of just chasing the tail.
        dv_x = self.plane.vel_x - self.missile.vel_x
        dv_y = self.plane.vel_y - self.missile.vel_y
        
        # Rotate relative velocity into the missile's local frame as well
        local_dv_x = dv_x * np.cos(theta) + dv_y * np.sin(theta)
        local_dv_y = -dv_x * np.sin(theta) + dv_y * np.cos(theta)

        # 6. Add heading error and speed difference
        target_angle = np.arctan2(dy, dx)
        heading_error = (target_angle - theta + np.pi) % (2 * np.pi) - np.pi
        heading_cos = np.cos(heading_error)
        heading_sin = np.sin(heading_error)
        speed_delta = self.plane.speed - self.missile.speed

        # Normalize observation values so learning is easier across wide distance/speed ranges.
        max_distance = 3500.0
        max_rel_speed = 400.0

        local_x = np.clip(local_x / max_distance, -1.0, 1.0)
        local_y = np.clip(local_y / max_distance, -1.0, 1.0)
        local_dv_x = np.clip(local_dv_x / max_rel_speed, -1.0, 1.0)
        local_dv_y = np.clip(local_dv_y / max_rel_speed, -1.0, 1.0)
        speed_delta = np.clip(speed_delta / max_rel_speed, -1.0, 1.0)

        # 7. Package it into a numpy array (Float32 for PyTorch compatibility)
        obs = np.array([
            local_x,          # Distance forward/backward (normalized)
            local_y,          # Distance left/right (normalized)
            local_dv_x,       # Closing speed forward/backward (normalized)
            local_dv_y,       # Closing speed left/right (normalized)
            heading_cos,      # Heading alignment cosine
            heading_sin,      # Heading alignment sine
            speed_delta       # Relative speed advantage (normalized)
        ], dtype=np.float32)
        
        return obs

    def _compute_reward(self, distance, terminated, truncated):

        if terminated:
            return 200.0
        if truncated:
            return -150.0

        progress = self.previous_distance - distance
        progress = np.clip(progress, -50.0, 50.0)
        
        # Kaçış cezası: eğer füze uzaklaşıyorsa ağır ceza
        escape_penalty = 0.0
        if progress < 0:
            escape_penalty = -0.8 * abs(progress)
        
        dist_penalty = 0.0018 * distance

        dx = self.plane.x - self.missile.x
        dy = self.plane.y - self.missile.y
        target_angle = np.arctan2(dy, dx)
        heading_error = (target_angle - self.missile.heading + np.pi) % (2 * np.pi) - np.pi
        heading_bonus = 0.25 * np.cos(heading_error)

        reward = 0.8 * progress - dist_penalty + heading_bonus + escape_penalty - 0.01
        return float(reward)