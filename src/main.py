#BU DOSYAYI TERMİMALDEN ÇALIŞTIRIN!! ÇÜNKÜ BORA'NIN LINUX SEVDASI YÜZÜNDEN PYTHON3 YÜZÜNDEN ÇALIŞMIYOR
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
    
    # Eğer hiçbir seçenek belirtilmemişse GUI başlat (varsayılan)
    if not (args.train or args.test):
        args.gui = True
    
    try:
        if args.train:
            print("🚀 Eğitim başlatılıyor...")
            from train import main as train_main
            # train_main() zaten --skip-test'i elle argparse'dan okur
            import os
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            # Eğer--skip-test geçildiyse, bunu train'e aktar
            sys.argv = ["train.py"]
            if args.skip_test:
                sys.argv.append("--skip-test")
            train_main()
            
        elif args.test:
            print("🧪 Test çalıştırılıyor...")
            from environment import MissileEnv
            from stable_baselines3 import PPO
            
            env = MissileEnv(render_mode="human")
            try:
                model = PPO.load("missile_ppo_model")
                obs, info = env.reset()
                for step in range(5000):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        obs, info = env.reset()
                print("✅ Test tamamlandı!")
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