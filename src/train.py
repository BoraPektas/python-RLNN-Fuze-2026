# train.py
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Kendi yazdığınız özel füze ortamını içeri aktarıyoruz
from environment import MissileEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-test", action="store_true", help="Skip the post-training interactive test run")
    args = parser.parse_args()

    # 1. Kendi yazdığımız Füze Ortamını (Environment) başlatıyoruz
    # render_mode=None: Eğitim sırasında optimized (imkansız ortamlar yok)
    env = MissileEnv(render_mode=None)
    # Monitor ile sararak SB3 için istatistiklerin kaydedilmesini sağlıyoruz
    env = Monitor(env)

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

    # 4. BORA BU OPSYONEL TEST ETMEK ZORUNDA DEĞİL, İSTERSEN YORUM SATIRI YAP: Eğitilen füzeyi test edip izleyelim
    # 4. BORA BU OPSYONEL TEST ETMEK ZORUNDA DEĞİL, İSTERSEN YORUM SATIRI YAP: Eğitilen füzeyi test edip izleyelim
    # 4. BORA BU OPSYONEL TEST ETMEK ZORUNDA DEĞİL, İSTERSEN YORUM SATIRI YAP: Eğitilen füzeyi test edip izleyelim
    # 4. BORA BU OPSYONEL TEST ETMEK ZORUNDA DEĞİL, İSTERSEN YORUM SATIRI YAP: Eğitilen füzeyi test edip izleyelim
    # 4. BORA BU OPSYONEL TEST ETMEK ZORUNDA DEĞİL, İSTERSEN YORUM SATIRI YAP: Eğitilen füzeyi test edip izleyelim
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL
    # TEKRAR EDİYORUM OPSYONEL

    if not args.skip_test:
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

        env_test.close()

    env.close()


if __name__ == "__main__":
    main()