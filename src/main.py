import numpy as np
import matplotlib.pyplot as plt
from environment import MissileEnv

def main(show_trajectories=True):
    env = MissileEnv()
    obs, info = env.reset()
    
    distances = []
    missile_xs = []
    missile_ys = []
    plane_xs = []
    plane_ys = []
    
    # Run a short simulation
    print("Running simulation...")
    for _ in range(500):
        # Taking random actions in continuous space (-1.0, 1.0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Access the distance
        distance = env.previous_distance
        distances.append(distance)
        
        # Store trajectories
        missile_xs.append(env.missile.x)
        missile_ys.append(env.missile.y)
        plane_xs.append(env.plane.x)
        plane_ys.append(env.plane.y)
        
        if terminated or truncated:
            print(f"Simulation ended early. Terminated: {terminated}, Truncated: {truncated}")
            break
            
    # Plot the results
    print("Simulation finished. Generating plot...")
    if show_trajectories:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot distance
        ax1.plot(distances, label="Distance to Target", color='blue')
        ax1.set_title("Distance between Missile and Plane")
        ax1.set_xlabel("Simulation Steps")
        ax1.set_ylabel("Distance")
        ax1.grid(True)
        ax1.legend()
        
        # Plot trajectories
        ax2.plot(missile_xs, missile_ys, label="Missile Path", color='red')
        ax2.plot(plane_xs, plane_ys, label="Plane Path", color='blue')
        
        # Mark start and end points
        ax2.scatter(missile_xs[0], missile_ys[0], color='red', marker='o', label="Missile Start")
        ax2.scatter(plane_xs[0], plane_ys[0], color='blue', marker='o', label="Plane Start")
        ax2.scatter(missile_xs[-1], missile_ys[-1], color='darkred', marker='x')
        ax2.scatter(plane_xs[-1], plane_ys[-1], color='darkblue', marker='x')
        
        ax2.set_title("Trajectories")
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Y Position")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(distances, label="Distance to Target", color='blue')
        plt.title("Distance between Missile and Plane over Time")
        plt.xlabel("Simulation Steps")
        plt.ylabel("Distance")
        plt.grid(True)
        plt.legend()
        
    plt.show()

if __name__ == "__main__":
    main(show_trajectories=True)