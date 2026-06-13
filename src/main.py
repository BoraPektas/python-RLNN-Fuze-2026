#!/usr/bin/env python3
"""
main.py: Project entry point — Launches the GUI only.
(True PN Architecture version: Provides the AI with LOS rate, closing velocity, etc.)
"""
import sys
import os
import time

def prompt_benchmark():
    try:
        model_name = input("\nDo you want to run the automated benchmark? (Type a model name e.g. PPO_6M.zip, or press Enter to skip and launch GUI): ").strip()
        if not model_name:
            return
            
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "..", "models", model_name)
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, model_name)
            if not os.path.exists(model_path):
                print(f"Could not find model '{model_name}'. Skipping benchmark.")
                return
                
        from stable_baselines3 import PPO
        from environment import MissileEnv
        
        print("\n" + "="*50)
        print(f"  AUTOMATED BENCHMARK (100 EPISODES)")
        print(f"  Model: {os.path.basename(model_path)}")
        print("="*50)
        print("Loading model and testing 100 random evasive environments headless. Please wait...")
        
        import warnings
        warnings.filterwarnings("ignore")
        
        model = PPO.load(model_path, device="cpu")
        env = MissileEnv(render_mode=None)
        
        hits = 0
        total_time = 0.0
        
        start_t = time.time()
        for i in range(100):
            obs, info = env.reset()
            done = False
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                steps += 1
                
            if term:
                hits += 1
                total_time += (steps * env.dt)
                
        print("\n--- BENCHMARK RESULTS ---")
        print(f"Total Episodes       : 100")
        print(f"Overall Hit Rate     : {hits}%")
        if hits > 0:
            print(f"Avg Interception Time: {total_time/hits:.2f} seconds")
        print(f"Computation Time     : {time.time() - start_t:.2f}s")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n[Benchmark Warning] Could not complete automated test: {e}\n")

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    prompt_benchmark()
    
    try:
        from gui import main_menu
        print("GUI Starting...")
        main_menu()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)