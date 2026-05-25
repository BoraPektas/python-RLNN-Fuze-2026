# train.py
import gymnasium as gym
from stable_baselines3 import PPO
# Kendi yazdığınız özel füze ortamını içeri aktarıyoruz
from environment import MissileEnv

# 1. Kendi yazdığımız Füze Ortamını (Environment) başlatıyoruz
# render_mode=None: Eğitim sırasında optimized (imkansız ortamlar yok)
env = MissileEnv(render_mode=None)

# 2. PyTorch tabanlı PPO algoritmasını bu ortama göre hazırlıyoruz
# "MlpPolicy" arka planda PyTorch sinir ağını otomatik oluşturur.
# Füzenin hareket alanı (action_space) sürekli (continuous) olduğu için PPO çok uygundur.
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# 3. Eğitimi Başlatıyoruz
# Füze simülasyonu CartPole'dan daha karmaşık olduğu için adım sayısını 50.000 yapalım
print("Füze Yapay Zekası Eğitimi Başlıyor...")
model.learn(total_timesteps=50000)
print("Eğitim tamamlandı!")

# Modelimizi daha sonra GUI (Arayüz) içinde direkt çağırıp kullanabilmek için kaydediyoruz
model.save("missile_ppo_model")
print("Model 'missile_ppo_model.zip' olarak kaydedildi.")

# 4. Eğitilen füzeyi test edip izleyelim
print("Eğitilen füze test ediliyor...")
# Yeni bir ortam oluştur (render_mode="human" ile tamamen rastgele ortamlar)
env_test = MissileEnv(render_mode="human")
obs, info = env_test.reset()
for _ in range(2000):
    # Eğitilen sinir ağı uçağın konumuna göre füzeye yön (sağ/sol) komutu verir
    action, _states = model.predict(obs, deterministic=True)
    
    # Komutu fizik motoruna gönder
    obs, reward, terminated, truncated, info = env_test.step(action)
    
    # Füze uçağı vurursa (terminated) veya gözden kaçırırsa (truncated) simülasyonu sıfırla
    if terminated or truncated:
        obs, info = env_test.reset()

env.close()
env_test.close()