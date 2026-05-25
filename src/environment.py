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

    def __init__(self, render_mode=None, truncation_mode="advanced"):
        super().__init__()
        self.render_mode = render_mode
        self.dt = 1.0 / 60.0  # Simulation time step
        
        # Truncation Mode: "simple" (old: distance > 3000) or "advanced" (new: missile distance from start > initial distance)
        self.truncation_mode = truncation_mode  # "simple" or "advanced"

        # 1. ACTION SPACE (What the AI can do)
        # Example: Continuous control of turning rate (angular velocity)
        # Values between -1.0 (max left) and 1.0 (max right)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 2. OBSERVATION SPACE (What the AI can see)
        # Example: [missile_x, missile_y, missile_angle, plane_x, plane_y, plane_angle]
        # You will likely want to normalize these inputs later or use relative coordinates.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.render_mode is None:
            # ===== EĞİTİM MODU: Kontrollü Rastgele Ortam =====
            # Füzenin her zaman uçağı yakalaması MÜMKÜN olacak şekilde setup yapılır
            # "Oransal Seyrüsefer" ilkesine göre:
            # - Füze hızı > Uçak hızı (300 > 150) ✓
            # - Başlangıç mesafesi yakın (200-400 birim)
            # - Relative geometry mantıklı (kaçış imkansız değil ama zor)
            
            missile_x = 0.0
            missile_y = 0.0
            
            # Uçak, füzenin etrafında 200-400 birim uzaklıkta rastgele
            distance = np.random.uniform(200, 400)
            angle = np.random.uniform(-np.pi, np.pi)
            plane_x = missile_x + distance * np.cos(angle)
            plane_y = missile_y + distance * np.sin(angle)
            
            # Füze MUTLAKA uçağa doğru bakmalı (180° tolerans ile)
            # Füze -> Uçak vektörü
            target_angle = np.arctan2(plane_y - missile_y, plane_x - missile_x)
            # Füze yönü: Hedef açı ± 45 derece (180° koşulu: ters yöne bakmamalı)
            missile_heading = target_angle + np.random.uniform(-np.pi/4, np.pi/4)  # ±45°
            missile_heading = (missile_heading + np.pi) % (2 * np.pi) - np.pi
            
            # Uçak yönü: +/- 90 derece arasında rastgele (kaçış geometrisi)
            plane_heading = missile_heading + np.random.uniform(-np.pi/2, np.pi/2)
            plane_heading = (plane_heading + np.pi) % (2 * np.pi) - np.pi
            
            missile_max_rotate = 2
            
        else:
            # ===== TEST MODU (render_mode="human"): Tamamen Rastgele Ortam =====
            # Simülasyonda imkansız durumlar da olabilir, bu normal
            plane_x = np.random.uniform(500, 1000)
            plane_y = np.random.uniform(500, 1000)
            plane_heading = np.random.uniform(-np.pi, np.pi)
            
            missile_x = 0.0
            missile_y = 0.0
            
            # Füze uçağa doğru bakmalı (test modunda da)
            target_angle = np.arctan2(plane_y - missile_y, plane_x - missile_x)
            missile_heading = target_angle + np.random.uniform(-np.pi/4, np.pi/4)
            missile_heading = (missile_heading + np.pi) % (2 * np.pi) - np.pi
            
            missile_max_rotate = 2
        
        # Store starting positions for miss detection (advanced mode)
        self.missile_start_x = missile_x
        self.missile_start_y = missile_y
        
        # Re-instantiate the classes (using your dynamic physics variables)
        self.plane = Plane(plane_x, plane_y, speed=150, heading=plane_heading, mass=500, thrust=200, drag=0.01, dt=self.dt)
        self.missile = Missile(missile_x, missile_y, speed=300, heading=missile_heading, mass=100, thrust=500, drag=0.005,max_rotate=missile_max_rotate, dt=self.dt)
        
        # Initialize the distance tracker for the reward function
        dist_x = self.plane.x - self.missile.x
        dist_y = self.plane.y - self.missile.y
        self.previous_distance = np.sqrt(dist_x**2 + dist_y**2)
        self.initial_plane_missile_distance = self.previous_distance  # Store for miss detection
        
        observation = self._get_obs()
        info = {}
        
        return observation, info

    def step(self, action):
        # 1. Update Physics
        self.missile.update(action)
        self.plane.update(rotate=0.2)

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
            # ADVANCED: Miss if missile traveled farther from start than initial plane-missile distance
            # Also keep the simple boundary check
            miss_condition = missile_dist_from_start > self.initial_plane_missile_distance
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

        # 6. Package it into a numpy array (Float32 for PyTorch compatibility)
        obs = np.array([
            local_x,          # Distance forward/backward
            local_y,          # Distance left/right
            local_dv_x,       # Closing speed forward/backward
            local_dv_y        # Closing speed left/right
        ], dtype=np.float32)
        
        return obs

    def _compute_reward(self, distance, terminated, truncated):

        reward = 0.0

        if terminated:
            # Huge bonus for the "Hit"
            reward = 100.0  
        elif truncated:
            # Penalty for "Lost Target", "Miss" or "Out of Bounds"
            reward = -50.0  
        else:
            # 1. SHAPING REWARD: Reward for closing the distance gap
            # progress > 0 means we got closer; progress < 0 means we drifted away
            progress = self.previous_distance - distance
            reward = progress * 0.1

            # 2. TIME PENALTY: Encourage the missile to hit as fast as possible
            # Without this, the AI might "circle" the target to farm progress rewards
            reward -= 0.01

        return reward