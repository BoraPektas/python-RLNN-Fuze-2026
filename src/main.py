#!/usr/bin/env python3
"""
main.py: Proje giriş noktası — GUI, eğitim, veya hızlı test seçeneği sunar.
"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="RLNN-Füze-2026: Pekiştirme Öğrenmeli Sinir Ağlı Güdümlü Füze",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py --gui              # Grafik arayüzü başlat
  python main.py --train            # Eğitim başlat
  python main.py --train --skip-test # Eğitim başlat (test etme)
  python main.py --test             # Hızlı test çalıştır
        """
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Grafik arayüzü (GUI) başlat (varsayılan)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Model eğitimini başlat"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Eğitim sonrası test adımını atla"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Hızlı test çalıştır (kaydedilmiş model gerekir)"
    )
    
    args = parser.parse_args()
    
    if not (args.train or args.test):
        args.gui = True
    
    try:
        if args.train:
            print("🚀 Eğitim başlatılıyor...")
            from train import main as train_main
            import os
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            sys.argv = ["train.py"]
            if args.skip_test:
                sys.argv.append("--skip-test")
            train_main()
            
        elif args.test:
            print("🧪 Test çalıştırılıyor...")
            from environment import MissileEnv
            from stable_baselines3 import PPO
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            
            env = MissileEnv(render_mode="human")
            try:
                model = PPO.load("missile_ppo_model")
                obs, info = env.reset()

                distances = []
                event_steps = []
                event_labels = []
                episodes = 0
                hits = 0
                misses = 0
                episode_trajectories = []
                current_missile_path = []
                current_plane_path = []

                max_steps = 25000
                target_episodes = 40
                step = 0
                episode_step = 0
                episode_lengths = []

                while episodes < target_episodes and step < max_steps:
                    distance = np.hypot(env.plane.x - env.missile.x, env.plane.y - env.missile.y)
                    distances.append(distance)
                    current_missile_path.append((env.missile.x, env.missile.y))
                    current_plane_path.append((env.plane.x, env.plane.y))

                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    episode_step += 1

                    if terminated or truncated:
                        episodes += 1
                        event = "hit" if terminated else "miss"
                        if terminated:
                            hits += 1
                        else:
                            misses += 1
                        event_steps.append(step)
                        event_labels.append(event)
                        episode_lengths.append(episode_step)
                        episode_trajectories.append({
                            "missile": current_missile_path,
                            "plane": current_plane_path,
                            "event": event,
                            "step": step,
                        })
                        current_missile_path = []
                        current_plane_path = []
                        obs, info = env.reset()
                        episode_step = 0

                if current_missile_path:
                    episode_trajectories.append({
                        "missile": current_missile_path,
                        "plane": current_plane_path,
                        "event": "end",
                        "step": step,
                    })
                    if episode_step > 0:
                        episode_lengths.append(episode_step)

                success_rate = 0.0
                if hits + misses > 0:
                    success_rate = hits / (hits + misses) * 100.0

                print(
                    f"✅ Test tamamlandı! Bölümler: {episodes}, vurma: {hits}, kaçırma: {misses}, başarı oranı: {success_rate:.1f}%"
                )

                # ===== YENİLENMİŞ MATPLOTLIB GÖRSELLEŞTİRME BLOGU =====
                plt.figure(figsize=(12, 10))
                
                # İlk Pas: Başarılı olanları soluk renkte arkaya çiziyoruz
                for episode in episode_trajectories:
                    if episode["event"] != "hit":
                        continue
                    missile_x = [p[0] for p in episode["missile"]]
                    missile_y = [p[1] for p in episode["missile"]]
                    plane_x = [p[0] for p in episode["plane"]]
                    plane_y = [p[1] for p in episode["plane"]]
                    
                    plt.plot(plane_x, plane_y, color="orange", linestyle="--", alpha=0.2, linewidth=0.8)
                    plt.plot(missile_x, missile_y, color="blue", alpha=0.3, linewidth=1.0)
                    plt.scatter(missile_x[-1], missile_y[-1], c="green", s=30, marker="x", alpha=0.4)

                # İkinci Pas: Iskalayan / Kaçan Rotaları en üste pürüzsüzce patlatıyoruz
                for episode in episode_trajectories:
                    if episode["event"] == "hit":
                        continue
                    missile_x = [p[0] for p in episode["missile"]]
                    missile_y = [p[1] for p in episode["missile"]]
                    plane_x = [p[0] for p in episode["plane"]]
                    plane_y = [p[1] for p in episode["plane"]]
                    
                    # Kaçan uçak kalın bordo, ıskalayan füze kalın kesikli kırmızı
                    plt.plot(plane_x, plane_y, color="darkred", linestyle="-.", alpha=0.9, linewidth=2.2)
                    plt.plot(missile_x, missile_y, color="red", linestyle="--", alpha=1.0, linewidth=2.5)
                    
                    # Kritik kaçış son konum işaretçileri
                    plt.scatter(missile_x[-1], missile_y[-1], c="black", s=130, marker="X", zorder=5)
                    plt.scatter(plane_x[-1], plane_y[-1], c="purple", s=110, marker="o", zorder=5)

                plt.title(f"2D Test Trajektoryası: {hits} vuruş / {episodes} bölüm ({success_rate:.1f}% başarı)\n(Kalın ve Kırmızı Çizgiler Iskalayan/Kaçırılan Rotaları Belirtir)", fontsize=12, fontweight='bold')
                plt.xlabel("X Pozisyonu")
                plt.ylabel("Y Pozisyonu")
                
                handles = [
                    Line2D([0], [0], color="orange", linestyle="--", alpha=0.4, label="Uçak Yolu (Başarılı)"),
                    Line2D([0], [0], color="blue", alpha=0.4, label="Füze Yolu (Başarılı)"),
                    Line2D([0], [0], color="darkred", linestyle="-.", linewidth=2, label="Uçak Yolu (KAÇTI!)"),
                    Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="Füze Yolu (KAÇIRDI!)"),
                    Line2D([0], [0], color="green", marker="x", linestyle="None", markersize=8, label="Başarılı Vuruş"),
                    Line2D([0], [0], color="black", marker="X", linestyle="None", markersize=10, label="Iskalama Noktası (Füze)"),
                    Line2D([0], [0], color="purple", marker="o", linestyle="None", markersize=8, label="Kurtulma Noktası (Uçak)"),
                ]
                plt.legend(handles=handles, loc="upper right")
                plt.grid(True, alpha=0.3)
                plt.axis("equal")
                plt.tight_layout()
                plt.show()

            except FileNotFoundError:
                print("❌ Hata: 'missile_ppo_model.zip' bulunamadı. Önce --train ile eğitim yapın.")
            finally:
                env.close()
                
        elif args.gui:
            print("🎮 GUI başlatılıyor...")
            from gui import main_menu
            main_menu()
            
    except KeyboardInterrupt:
        print("\n⏹️  Kullanıcı tarafından durduruldu.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()