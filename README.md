## python-RLNN-Fuze-2026
2025-2026 Dönemi Python Programlamaya Giriş Dersi: Pekiştirmeli Öğrenme Sinir Ağlı Güdümlü Füze.

---

## İçindekiler
- [Kurulum](#kurulum)
- [Kullanım](#kullanim)
- [Proje Yapısı](#proje-yapısı)
- [Teknik Özellikler](#teknik-özellikler)
- [Katkılar](#katkılar)

---

## Kurulum

### 1. Repo'yu Klonlayın
```bash
git clone https://github.com/BoraPektas/python-RLNN-Fuze-2026.git
cd python-RLNN-Fuze-2026
```

### 2. Sanal Ortamı Kurun

**Windows:**
1. Python 3.12.9'u indirin. Kurulum sırasında "Add Python to PATH" seçeneğini işaretlediğinizden emin olun. Eğer bilgisayarınızda birden fazla Python sürümü varsa, spesifik olarak 3.12 kullanmak için Py Launcher (`py`) kullanacağız.
2. Sanal ortamı oluşturun:
   ```bash
   py -3.12 -m venv venv
   ```
3. Sanal ortamı aktifleştirin:
   ```bash
   .\venv\Scripts\activate
   ```
4. Sanal ortamın doğru sürümle açıldığını teyit edin:
   ```bash
   python --version
   ```

**Linux / macOS:**
1. pyenv gibi bir versiyon aracı ile bilgisayarınıza Python 3.12.9 kurun.
2. Shell üzerinde 3.12.9 sürümünü aktifleştirin ve doğrulayın:
   ```bash
   pyenv shell 3.12.9
   python3 --version
   ```
3. Sanal ortamı oluşturun:
   ```bash
   python3 -m venv venv
   ```
4. Sanal ortamı aktifleştirin:
   ```bash
   source venv/bin/activate
   ```
5. Pano (clipboard) desteği için xclip'i indirin:
   - Debian/Ubuntu: `sudo apt-get install xclip`
   - Arch Linux: `sudo pacman -S xclip`
   - Fedora: `sudo dnf install xclip`

### 3. Gerekli Kütüphaneleri Kurun
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Kullanım

Bu proje tamamen Grafik Kullanıcı Arayüzü (GUI) üzerinden yönetilmektedir. Doğrudan CLI komutlarına ihtiyaç yoktur.

Arayüzü başlatmak için:

**Windows:**
```bash
python src\main.py
```

**Linux / macOS:**
```bash
python3 src/main.py
```

### Model Eğitimi
1. Ana menüden "Train AI" seçeneğine tıklayın.
2. Ekranda görmek istediğiniz "Total Steps" (Toplam Adım) miktarını belirleyin. Standart bir eğitim için 5,000,000 adım önerilir.
3. Eğitimi başlatmak için "Start Training" butonuna basın ve modelin kaydedileceği hedef dizini seçin.
4. Eğitim işlemi arka planda PPO algoritmasını 16 paralel CPU ortamında çalıştıracaktır. Ekrandaki durum çubuğundan ilerlemeyi takip edebilirsiniz.

### Simülasyon Kullanımı
1. Ana menüden "Simulation" seçeneğine tıklayın.
2. Sağ menüden "Load Model" seçeneği ile eğitmiş olduğunuz bir .zip modelini seçin.
3. Ekranda uçak ve füze konumlarını sürükleyerek veya açılarını döndürerek başlangıç geometrisini ayarlayın.
4. "PLAY" butonuna basarak yapay zekanın performansını izleyebilir, veya "PLAY PN" butonuna basarak yapay zekayı geleneksel Oransal Yönlendirme (Proportional Navigation) algoritması ile kıyaslayabilirsiniz.

---

## Proje Yapısı

```text
python-RLNN-Fuze-2026/
├── src/
│   ├── main.py              # Proje giriş noktası, arayüzü başlatır
│   ├── environment.py       # 2D Fizik motoru ve RL (Gym) eğitim ortamı
│   ├── train.py             # PPO Eğitim pipeline'ı (Arka planda çalışır)
│   └── gui.py               # Pygame tabanlı grafik kullanıcı arayüzü
├── models/                  # Eğitilmiş sinir ağı (.zip) modellerinin dizini
├── requirements.txt         # Proje bağımlılıkları (PyTorch, Stable-Baselines3, vb.)
├── train_progress.json      # Eğitim sürecinin anlık durumunu arayüze aktarır
├── README.md                # Bu döküman
└── LICENSE                  # Proje lisansı
```

---

## Teknik Özellikler

### Fizik Motoru ve Ortam
- **2D Dinamik Model:** Füze ve uçağın hız, ivme, itki ve sürtünme kuvvetleri üzerinden gerçekçi takibi. Sadece kinematik değil, kütle-sürtünme denklemleri çözümlenmektedir.
- **Parametrik Rastgelelik:** Ağın genellenebilirliğini (generalization) artırmak için her iterasyonda füze kütlesi (50-120), hızı (350-550), itkisi, uçak hızı ve ağırlığı randomize edilir. Geometri de her seferinde yeniden kurgulanır.
- **Kaçış Manevraları:** Hedef uçak eğitim boyunca sabit uçuş, dar dairesel loiter ve yüksek-G harmonik (sinusoidal) kaçış olmak üzere üç farklı davranış kalıbı sergiler.
- **Fizibilite Filtresi (Feasibility Test):** Eğitim ortamı oluşturulurken geometri, klasik oransal yönlendirme formülü ile matematiksel olarak arka planda test edilir. Sadece füzenin teorik olarak yetişebileceği "fizibl" geometriler yapay zekaya sunulur. Yakalanamaz durumlar `tail_chase_failure` gibi optimizasyonlarla başlangıçta filtrelenir.

### Pekiştirmeli Öğrenme (Reinforcement Learning) Mimarisi
- **Algoritma:** PyTorch tabanlı Stable-Baselines3 kütüphanesi üzerinden `PPO` (Proximal Policy Optimization) kullanılmıştır.
- **Girdi Uzayı (Observation Space):** Ağ, Kartezyen koordinat sisteminden tamamen izole edilmiştir. Sadece Oransal Yönlendirme mantığı için kritik olan iki parametreyi girdi olarak alır:
  1. `Görüş Hattı Hızı (LOS Rate)`
  2. `Yaklaşma Hızı (Closing Velocity - Vc)`
- **Ödül Fonksiyonu (Reward Function):** İki boyutlu girdi avantajı sayesinde, sistem hedefe yaklaşmayı ödüllendiren (dense reward) ve gecikmeyi cezalandıran basit bir yapıya oturtulmuştur. Ağ, Pure Pursuit (Saf Takip) tuzağına düşmeden Oransal Yönlendirme davranışını kendiliğinden keşfetmektedir.
- **Paralelizasyon:** Eğitim verimini ve veri akışını maksimize etmek adına `SubprocVecEnv` kullanılarak CPU üzerinde 16 paralel environment aynı anda işlenir.

---

## Kullanılan Kütüphaneler ve Teknolojiler
Projeyi geliştirirken kullandığımız temel kütüphaneler ve görevleri şunlardır:

- **PyTorch (`torch`)**: Tüm yapay zeka modelinin (PPO) arka planında hesaplamaları ve optimizasyonları (tensor işlemleri, geri yayılım) yapan temel Derin Öğrenme kütüphanesidir.
- **Stable-Baselines3 (`stable_baselines3`)**: PPO (Proximal Policy Optimization) takviyeli öğrenme algoritmasının güvenilir ve güncel implementasyonunu sağlar. Model mimarisi ve çoklu işlem (multiprocessing) altyapısı buradan çekilmiştir.
- **Gymnasium (`gymnasium`)**: Kendi yazdığımız 2D fizik motorumuzu (`MissileEnv`), RL algoritmalarının anlayabileceği evrensel standartlara oturtmak için kullandığımız RL ortamı framework'üdür.
- **Pygame (`pygame`)**: Projenin tamamen interaktif olan Grafik Kullanıcı Arayüzünü (GUI) yönetir. Ana menülerden, eğitim sırasındaki canlı yükleme barına ve 2 boyutlu uzaysal simülasyonun ekrana çizilmesine kadar her türlü görsel render işlemi bu kütüphaneyle yapılmıştır.
- **NumPy (`numpy`)**: Fizik motoru içindeki ağır matematiksel yükü üstlenir. Trigonometrik manevra hesaplamaları (sin, cos, radyan dönüşümleri), hız vektörlerinin izdüşümü ve yapay zekaya sunulan çok boyutlu tensor verilerinin hızlıca işlenmesi için kullanılmıştır.

---

## Katkılar
- **031890087** Ahmet Şeref Gölcük
- **032490011** Efe Hüner
- **032490028** Bora Pektaş

---

## Lisans
MIT License (bkz. LICENSE dosyası)
