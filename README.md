## RLNN-Fuze-2026
2026-2027 Dönemi Python Programlamaya Giriş Dersi: Pekiştirme Öğrenmeli Sinir Ağlı Güdümlü Füze.

## Kurmak

 1. Repo'yu Klonlayın
```bash
git clone https://github.com/BoraPektas/RLNN-Fuze-2026.git
cd RLNN-Fuze-2026
```
 2. Sanal Ortam'ı Kurun
	 Windows:
	 1. [Python 3.12.9](https://www.python.org/downloads/release/python-3129/)'u indirin. Otomatik PATH kaydı oluştuğundan emin olun
	 2.  Sanal ortamı oluşturun
	 `python -m venv venv`
	 3. Sanal ortamı aktifleştirin
	 `.\venv\Scripts\activate`
	 
	 Linux:
	 1. En güncel olmayan 3.12.9 sürümü kullanıldığı için [pyenv](https://github.com/pyenv/pyenv) gibi bir versiyon aracı ile bilgisayarınıza Python 3.12.9 kurun.
	 2.  Sanal ortamı oluşturun
	 `python -m venv venv`
	 3. Sanal ortamı aktifleştirin
	 `source venv/bin/activate`
	 4. xclip'i indirin
	 debian: `sudo apt-get install xclip` arch:`sudo pacman -S xclip`
3. Sanal ortam oluşunca gerekli kütüphaneleri kurun
`pip install --upgrade pip`
`pip install -r requirements.txt`


