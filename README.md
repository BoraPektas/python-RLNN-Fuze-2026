## RLNN-Fuze-2026
2026-2027 Dönemi Python Programlamaya Giriş Dersi: Pekiştirme Öğrenmeli Sinir Ağlı Güdümlü Füze.

---

## 📋 İçindekiler
- [Kurmak](#kurmak)
- [Çalıştırma](#çalıştırma)
- [Proje Yapısı](#proje-yapısı)
- [Özellikler](#özellikler)
- [Geliştirme Notları](#geliştirme-notları)

---

## 🛠️ Kurmak

### 1. Repo'yu Klonlayın
```bash
git clone https://github.com/BoraPektas/RLNN-Fuze-2026.git
cd RLNN-Fuze-2026
```

### 2. Sanal Ortam'ı Kurun

**Windows:**
1. [Python 3.12.9](https://www.python.org/downloads/release/python-3129/)'u indirin. Otomatik PATH kaydı oluştuğundan emin olun
2. Sanal ortamı oluşturun:
   ```bash
   python -m venv venv
   ```
3. Sanal ortamı aktifleştirin:
   ```bash
   .\venv\Scripts\activate
   ```

**Linux / macOS:**
1. En güncel olmayan 3.12.9 sürümü kullanıldığı için [pyenv](https://github.com/pyenv/pyenv) gibi bir versiyon aracı ile bilgisayarınıza Python 3.12.9 kurun.
2. Sanal ortamı oluşturun:
   ```bash
   python -m venv venv
   ```
3. Sanal ortamı aktifleştirin:
   ```bash
   source venv/bin/activate
   ```
4. xclip'i indirin (clipboard desteği için):
   - Debian: `sudo apt-get install xclip`
   - Arch: `sudo pacman -S xclip`

### 3. Gerekli Kütüphaneleri Kurun
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Çalıştırma

### Grafik Arayüzü (GUI) Başlat
```bash
cd src
python main.py
```
veya doğrudan:
```bash
cd src
python main.py --gui
```

### Eğitim Başlat
```bash
cd src
python main.py --train
```

Eğitim sonrası test adımını atlamak için:
```bash
cd src
python main.py --train --skip-test
```

### Hızlı Test (Kaydedilmiş Model)
```bash
cd src
python main.py --test
```

---

## 📁 Proje Yapısı

```
python-RLNN-Fuze-2026/
├── src/
│   ├── main.py              # Giriş noktası (CLI / GUI seçer)
│   ├── environment.py       # Fizik motoru ve Gym ortamı
│   ├── train.py             # Eğitim pipeline'ı (PPO)
│   ├── gui.py               # Grafik arayüzü (Pygame)
│   └── missile_ppo_model    # Kaydedilmiş model (eğitimden sonra oluşur)
├── requirements.txt         # Python bağımlılıkları
├── README.md               # Bu dosya
└── LICENSE                 # Proje lisansı
```

---

## ✨ Özellikler

### 🎮 Fizik Motoru
- **2D Dinamikler**: Füze ve uçağın konumu, hızı, yönü, kuvvetleri (itme, sürtünme)
- **Rastgele Ortam Üretimi**: Her eğitim adımında rastgele füze/uçak fizik parametreleri
- **Feasibility Testi**: Ortam oluşturulurken oransal yönlendirme ile test edilir — vurursa eğitime alınır
- **Adım-Başına Kontrol**: Uçağa sabit dönüş, callable, veya string ifade (örn. `"0.2 * sin(t)"`) geçebilirsiniz

### 🤖 Eğitim
- **PPO (Proximal Policy Optimization)**: Stable-Baselines3 kütüphanesi kullanıyor
- **Ödül Fonksiyonu**: Mesafe kapatma, zaman cezası, vuruş bonusu
- **Gözlem Uzayı**: Göreceli konum ve hız (füzenin yerel çerçevesi)

### 🎨 Grafik Arayüzü (GUI)
- **Ana Menü**: Eğitim, Simülasyon, Hakkında ekranlarına erişim
- **Eğitim Ekranı**: Eğitim başlat / durdur, canlı durum takibi
- **Simülasyon Ekranı**: İnteraktif editor — füze/uçak konumları ve açıları ayarlanabilir, özel alanlar eklenebilir
- **Temel Fizik Görselleştirmesi**: Harita, zoom, pan, grid

### 🔄 Oransal Yönlendirme (Feasibility Testi)
- Her rastgele ortam bir oransal yönlendirme kontrolörü ile test edilir
- Eğer füze uçağı yakalayabiliyor → ortam eğitime alınır
- Eğer yakalayamıyor → yeni rastgele parametre dener (max 6 deneme)

---

## 🧪 Geliştirme Notları

### Ortam API

#### Temel Kullanım
```python
from src.environment import MissileEnv
env = MissileEnv(render_mode=None)  # Eğitim modu
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

#### Uçak Kontrolü Ayarlama
```python
# Sabit dönüş hızı (0.2 rad/s)
env.set_plane_control(0.2)

# Sinüsoid kontrol: sin(t) şeklinde daire çizer
env.set_plane_control("0.5 * sin(t)")

# Python fonksiyonu
def ctrl(t):
    return 0.3 * np.sin(2.0 * t)
env.set_plane_control(ctrl)

# Reset sırasında ayarla
obs, info = env.reset(options={"plane_control": "0.2"})
```

### Eğitim Parametreleri
`train.py` içinde ayarlanabilir:
- `total_timesteps`: Toplam adım sayısı (varsayılan: 50000)
- `learning_rate`: Öğrenme hızı (varsayılan: 0.0003)

### Sorun Giderme

**Hata: `FileNotFoundError: missile_ppo_model not found`**
- Önce `python main.py --train` ile eğitim yapın

**GUI'de şrift sorunları**
- Pygame varsayılan şriftleri kullanır (`Consolas`). Sisteminizde yoksa `pygame.font.SysFont("arial", ...)` ile değiştirin

**Eğitim çok yavaş**
- `train.py`'de `total_timesteps` küçültün veya GPU/CUDA kurulumunu kontrol edin

---

## 👨‍💻 Katkılar
- **031890087** Ahmet Şeref Gölcük
- **032490011** Efe Hüner
- **032490028** Bora Pektaş

---

## 📄 Lisans
MIT License (bkz. [LICENSE](LICENSE) dosyası)
