# train.py
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Kendi yazdığınız özel füze ortamını içeri aktarıyoruz
from environment import MissileEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-test", action="store_true", help="Skip the post-training interactive test run")
    args = parser.parse_args()

    # 1. Kendi yazdığımız Füze Ortamını (Environment) başlatıyoruz
    # render_mode=None: Eğitim sırasında optimized (imkansız ortamlar yok)
    env = DummyVecEnv([lambda: Monitor(MissileEnv(render_mode=None))])

    # 2. PyTorch tabanlı PPO algoritmasını bu ortama göre hazırlıyoruz
    # "MlpPolicy" arka planda PyTorch sinir ağını otomatik oluşturur.
    # Füzenin hareket alanı (action_space) sürekli (continuous) olduğu için PPO çok uygundur.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        target_kl=0.02,
        n_epochs=10,
        device="cuda",
        policy_kwargs={"net_arch": [dict(pi=[256, 256], vf=[256, 256])]},
    )

    # 3. Eğitimi Başlatıyoruz
    # GPU ile hızlı test amaçlı eğitim
    print("Füze Yapay Zekası Eğitimi Başlıyor... (200000 timesteps, GPU ile)")
    model.learn(total_timesteps=300000)
    print("Eğitim tamamlandı!")

    # Modelimizi daha sonra GUI (Arayüz) içinde direkt çağırıp kullanabilmek için kaydediyoruz
    model.save("missile_ppo_model")
    print("Model 'missile_ppo_model.zip' olarak kaydedildi.")

    env.close()


if __name__ == "__main__":
    main()