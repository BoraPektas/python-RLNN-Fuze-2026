#gui.py görsel kullanıcı arayüzünün çalıştığı yerdir.
import pygame
import sys
import pyperclip
import webbrowser
import math
import subprocess
import os
import tkinter as tk
from tkinter import filedialog
import copy
import numpy as np

# Yapay Zeka ve Fizik Motoru Entegrasyonları
try:
    from stable_baselines3 import PPO
    from environment import MissileEnv
except ImportError:
    print("Uyarı: stable_baselines3 veya environment modülü bulunamadı. Simülasyon başlatılamayabilir.")

pygame.init()

BACKGROUND_COLOR = (180, 180, 180)
PANEL_COLOR      = (50, 50, 50)
BUTTON_COLOR     = (120, 120, 120)
TEXT_COLOR       = (20, 20, 20)
LINK_COLOR       = (70, 130, 180)
WHITE            = (255, 255, 255)

text_font    = pygame.font.SysFont("Consolas", 50, bold=True)
link_font    = pygame.font.SysFont("Consolas", 32, bold=True)
caption_font = pygame.font.SysFont("Consolas", 100, bold=True)
small_font   = pygame.font.SysFont("Consolas", 20, bold=True)

screen = pygame.display.set_mode((1000, 750))


def draw_text(text, font, color, surface, x, y):
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(x, y))
    surface.blit(surf, rect)


# ─── MAIN MENU ────────────────────────────────────────────────────────────────
def main_menu():
    pygame.display.set_caption('Missile RLNN - Main Menu')
    while True:
        screen.fill(BACKGROUND_COLOR)
        mp = pygame.mouse.get_pos()

        draw_text("MAIN MENU", caption_font, TEXT_COLOR, screen, 500, 150)
        pygame.draw.line(screen, TEXT_COLOR, (250, 210), (750, 210))

        bw, bh = 450, 70
        bx = 500 - bw // 2
        btn_sim   = pygame.Rect(bx, 250, bw, bh)
        btn_train = pygame.Rect(bx, 350, bw, bh)
        btn_about = pygame.Rect(bx, 450, bw, bh)
        btns = [btn_sim, btn_train, btn_about]

        for rect, label in [(btn_sim, "SIMULATE MODEL"),
                            (btn_train, "TRAIN MODEL"),
                            (btn_about, "ABOUT")]:
            c = (130,130,130) if rect.collidepoint(mp) else BUTTON_COLOR
            pygame.draw.rect(screen, c, rect)
            pygame.draw.rect(screen, TEXT_COLOR, rect, 3)
            draw_text(label, text_font, WHITE, screen, rect.centerx, rect.centery)

        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if any(r.collidepoint(mp) for r in btns)
                                else pygame.SYSTEM_CURSOR_ARROW)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_sim.collidepoint(mp):   simulate_screen()
                if btn_train.collidepoint(mp): train_screen()
                if btn_about.collidepoint(mp): about_screen()

        pygame.display.update()


# ─── TRAIN MODEL EKRANI ───────────────────────────────────────────────────────
def train_screen():
    running = True
    clock = pygame.time.Clock() 
    
    btn_w, btn_h = 450, 70
    bx = 500 - btn_w // 2
    
    btn_start_train = pygame.Rect(bx, 280, btn_w, btn_h)
    btn_back        = pygame.Rect(bx, 420, btn_w, btn_h)
    
    status_message = "Bekleniyor..."
    status_color = (50, 50, 150)

    is_training = False
    training_process = None
    progress_value = 0.0

    while running:
        screen.fill(BACKGROUND_COLOR)
        mp = pygame.mouse.get_pos()

        draw_text('TRAIN MODEL', caption_font, TEXT_COLOR, screen, 500, 120)

        for rect, label in [(btn_start_train, "START TRAINING"),
                            (btn_back, "BACK")]:
            c = (130, 130, 130) if rect.collidepoint(mp) else BUTTON_COLOR
            pygame.draw.rect(screen, c, rect, border_radius=8)
            pygame.draw.rect(screen, TEXT_COLOR, rect, 3, border_radius=8)
            draw_text(label, text_font, WHITE, screen, rect.centerx, rect.centery)

        draw_text(f"Durum: {status_message}", small_font, status_color, screen, 500, 520)

        # ─── YÜKLEME BARI (PROGRESS BAR) ANİMASYONU ───
        if is_training:
            if training_process and training_process.poll() is not None:
                is_training = False 
                status_message = "Eğitim Tamamlandı! (.zip oluşturuldu)"
                status_color = (0, 150, 0)
            else:
                bar_w = 450
                bar_h = 30
                bar_x = 500 - bar_w // 2
                bar_y = 600 

                progress_value += 1.5 
                if progress_value > 100:
                    progress_value = 0

                pygame.draw.rect(screen, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h), border_radius=5)
                
                fill_w = int((progress_value / 100) * bar_w)
                if fill_w > 0:
                    pygame.draw.rect(screen, (40, 200, 40), (bar_x, bar_y, fill_w, bar_h), border_radius=5)
                
                pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=5)
                
                draw_text("Eğitiliyor, lütfen bekleyin...", small_font, (60, 60, 60), screen, 500, bar_y + 45)

        if any(r.collidepoint(mp) for r in [btn_start_train, btn_back]):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_back.collidepoint(mp):
                    running = False
                
                elif btn_start_train.collidepoint(mp):
                    if not is_training: 
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes('-topmost', True)
                        
                        save_path = filedialog.asksaveasfilename(
                            title="Modeli Nereye Kaydedelim? (.zip)",
                            defaultextension=".zip",
                            filetypes=[("Zip Dosyaları", "*.zip"), ("Tüm Dosyalar", "*.*")]
                        )
                        root.destroy()
                        
                        if save_path:
                            is_training = True
                            progress_value = 0.0
                            status_message = "Eğitim başlatıldı! (Arka planda çalışıyor...)"
                            status_color = (0, 150, 0)
                            
                            try:
                                train_path = os.path.join(os.path.dirname(__file__), 'train.py')
                                training_process = subprocess.Popen([sys.executable, train_path, save_path])
                            except Exception as e:
                                status_message = f"Hata: {str(e)}"
                                status_color = (200, 0, 0)
                                is_training = False
                        else:
                            status_message = "Kaydetme iptal edildi. Eğitim başlamadı."
                            status_color = (150, 50, 50)

        pygame.display.update()
        clock.tick(60)


# ─── ABOUT ────────────────────────────────────────────────────────────────────
def about_screen():
    running  = True
    repo_url = 'https://github.com/BoraPektas/RLNN-Fuze-2026'
    lw, lh   = 800, 40
    link_rect = pygame.Rect(500 - lw//2, 500 - lh//2, lw, lh)
    copy_timer = 0

    while running:
        screen.fill(BACKGROUND_COLOR)
        mp = pygame.mouse.get_pos()
        pygame.display.set_caption('About')

        draw_text('ABOUT', caption_font, TEXT_COLOR, screen, 500, 100)
        for text, y in [('Missile RLNN  Project', 200),
                        ('031890087 Ahmet Şeref Gölcük', 300),
                        ('032490011 Efe Hüner', 350),
                        ('032490028 Bora Pektaş', 400)]:
            draw_text(text, text_font, TEXT_COLOR, screen, 500, y)

        hover = link_rect.collidepoint(mp)
        lc = (100,160,220) if hover else LINK_COLOR
        draw_text(repo_url, link_font, lc, screen, 500, 500)
        if not hover:
            pygame.draw.line(screen, lc, (110,520),(890,520), 2)
        if copy_timer > 0:
            draw_text('Link coppied to clipboard', small_font, (0,120,0), screen, 500, 550)
            copy_timer -= 1

        btn_back = pygame.Rect(700, 600, 200, 60)
        bc = (130,130,130) if btn_back.collidepoint(mp) else BUTTON_COLOR
        pygame.draw.rect(screen, bc, btn_back)
        pygame.draw.rect(screen, TEXT_COLOR, btn_back, 3)
        draw_text("BACK", text_font, WHITE, screen, btn_back.centerx, btn_back.centery)

        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if (link_rect.collidepoint(mp) or btn_back.collidepoint(mp))
                                else pygame.SYSTEM_CURSOR_ARROW)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_back.collidepoint(mp): running = False
                if link_rect.collidepoint(mp):
                    webbrowser.open(repo_url); pyperclip.copy(repo_url); copy_timer = 180

        pygame.display.update()


# ─── SIMULATE ─────────────────────────────────────────────────────────────────
def simulate_screen():
    running = True
    clock = pygame.time.Clock()

    camera_x, camera_y = 0.0, 0.0
    zoom = 1.0

    entities       = {}
    dragging_type  = None
    current_drag_angle = 0
    is_panning     = False
    last_mouse_pos = pygame.mouse.get_pos()
    active_edit    = None   

    PANEL_W    = 300
    PANEL_X    = 1000 - PANEL_W   
    PANEL_H    = 750
    PANEL_RECT = pygame.Rect(PANEL_X, 0, PANEL_W, PANEL_H)

    ICON_AREA_H = 255   
    ESC_AREA_H  = 40
    HUD_TOP     = ICON_AREA_H
    HUD_BOT     = PANEL_H - ESC_AREA_H
    HUD_H       = HUD_BOT - HUD_TOP

    # ── SİMÜLASYON DURUMU VE YAPAY ZEKA DEĞİŞKENLERİ ──
    is_playing = False
    initial_entities_state = {} 
    
    loaded_model = None
    loaded_model_path = None
    env = None
    current_obs = None

    btn_model = pygame.Rect(PANEL_X + 10, 8, PANEL_W - 20, 28)
    
    # ── KONTROL BUTONLARI (PLAY/STOP/RESET) ──
    btn_play  = pygame.Rect(20, 20, 100, 40)
    btn_stop  = pygame.Rect(130, 20, 100, 40)
    btn_reset = pygame.Rect(240, 20, 100, 40)

    ICON_SIZE = 60
    half_w    = PANEL_W // 2  

    plane_cx   = PANEL_X + half_w // 2
    plane_cy   = 130
    plane_icon = pygame.Rect(plane_cx - ICON_SIZE//2, plane_cy - ICON_SIZE//2, ICON_SIZE, ICON_SIZE)

    missile_cx = PANEL_X + half_w + half_w // 2
    missile_cy = 130
    missile_icon_rect   = pygame.Rect(missile_cx - ICON_SIZE//2, missile_cy - ICON_SIZE//2, ICON_SIZE, ICON_SIZE)
    missile_icon_center = (missile_cx, missile_cy)

    hud_scroll    = 0
    hud_content_h = 0
    MAX_SCROLL    = 0
    hud_surface = pygame.Surface((PANEL_W, 4000))
    hud_clickables = []  

    def draw_rotated_entity(surface, et, cx, cy, size, angle):
        half = size/2
        pts  = ([(-half,-half),(half,-half),(half,half),(-half,half)]
                if et=='plane' else [(0,-half),(-half,half),(half,half)])
        color= (0,180,255) if et=='plane' else (220,20,20)
        rad  = math.radians(-angle)
        ca,sa= math.cos(rad), math.sin(rad)
        rpts = [(cx+px*ca-py*sa, cy+px*sa+py*ca) for px,py in pts]
        pygame.draw.polygon(surface, color, rpts)
        if et=='plane':
            pygame.draw.polygon(surface, (0,100,200), rpts, 2)

    def drag_world_pos(mpos):
        return (mpos[0]-camera_x)/zoom, (mpos[1]-camera_y)/zoom

    ROW_H   = 28
    FIELD_W = PANEL_W - 16

    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        # ── YAPAY ZEKA (AI) FİZİK MOTORU DÖNGÜSÜ ──
        if is_playing and loaded_model is not None and env is not None:
            # 1. Modelden kararı (action) iste
            action, _ = loaded_model.predict(current_obs, deterministic=True)
            
            # 2. Fizik motorunda (environment.py) 1 frame ilerle
            current_obs, reward, terminated, truncated, info = env.step(action)
            
            # 3. Fizik motorundan dönen yeni koordinatları GUI'ye güncelle
            entities['plane']['x'] = env.plane.x
            entities['plane']['y'] = env.plane.y
            entities['plane']['angle'] = -math.degrees(env.plane.heading)
            
            entities['missile']['x'] = env.missile.x
            entities['missile']['y'] = env.missile.y
            entities['missile']['angle'] = -math.degrees(env.missile.heading)
            
            # 4. Çarpışma veya alandan çıkma olursa simülasyonu durdur
            if terminated or truncated:
                is_playing = False

        # ── GRID + ARKA PLAN ──────────────────────────────────────────
        screen.fill(WHITE)

        wl=-camera_x/zoom; wt=-camera_y/zoom
        wr=(PANEL_X-camera_x)/zoom; wb=(PANEL_H-camera_y)/zoom

        for wy in range(int(wt//50)*50, int(wb)+50, 50):
            sy=wy*zoom+camera_y
            pygame.draw.line(screen, (120,120,120) if wy%500==0 else (220,220,220), (0,sy),(PANEL_X,sy), 3 if wy%500==0 else 1)

        for wx in range(int(wl//50)*50, int(wr)+50, 50):
            sx=wx*zoom+camera_x
            pygame.draw.line(screen, (120,120,120) if wx%500==0 else (220,220,220), (sx,0),(sx,PANEL_H), 3 if wx%500==0 else 1)

        # ── NESNELER ──────────────────────────────────────────────────
        size=60*zoom
        for et,data in entities.items():
            if et == dragging_type:
                continue

            sx=data['x']*zoom+camera_x; sy=data['y']*zoom+camera_y
            if sx-size/2>PANEL_X: continue
            draw_rotated_entity(screen,et,sx,sy,size,data['angle'])
            
        # ── KONTROL BUTONLARINI ÇİZME ──
        for rect, label, base_color in [(btn_play, "PLAY", (60, 160, 60)), 
                                        (btn_stop, "STOP", (180, 60, 60)), 
                                        (btn_reset, "RESET", (180, 160, 60))]:
            if (label == "PLAY" and is_playing) or (label == "STOP" and not is_playing):
                c = (base_color[0]//2, base_color[1]//2, base_color[2]//2) # Daha koyu
            else:
                c = (200, 200, 200) if rect.collidepoint(mouse_pos) else base_color
                
            pygame.draw.rect(screen, c, rect, border_radius=5)
            pygame.draw.rect(screen, (30,30,30), rect, 2, border_radius=5)
            draw_text(label, small_font, WHITE if c != (200,200,200) else TEXT_COLOR, screen, rect.centerx, rect.centery)


        # ── SAĞ PANEL ─────────────────────────────────────────────────
        pygame.draw.rect(screen, PANEL_COLOR, PANEL_RECT)
        pygame.draw.line(screen,(0,0,0),(PANEL_X,0),(PANEL_X,PANEL_H),3)

        # ── MODEL SEÇ butonu ──────────────────────────────────────────
        mp_hover = btn_model.collidepoint(mouse_pos)
        btn_bg   = (80,130,80)  if mp_hover else (55,90,55)
        btn_bc   = (120,200,120) if mp_hover else (80,140,80)
        pygame.draw.rect(screen, btn_bg, btn_model, border_radius=4)
        pygame.draw.rect(screen, btn_bc, btn_model, 1, border_radius=4)
        if loaded_model_path:
            import os
            short = os.path.basename(loaded_model_path)
            if len(short) > 22: short = short[:19] + "..."
            btn_label = f"✓ {short}"
            lbl_color = (180,255,180)
        else:
            btn_label = "SELECT MODEL"
            lbl_color = (200,255,200)
        lbl_surf = small_font.render(btn_label, True, lbl_color)
        lbl_rect = lbl_surf.get_rect(center=btn_model.center)
        screen.blit(lbl_surf, lbl_rect)
        if mp_hover:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)

        draw_text("ENTITIES", small_font, (180,180,180), screen, PANEL_X + PANEL_W//2, plane_cy - ICON_SIZE//2 - 18)

        mid_x = PANEL_X + half_w
        pygame.draw.line(screen, (80,80,80), (mid_x, plane_cy - ICON_SIZE//2 - 6), (mid_x, plane_cy + ICON_SIZE//2 + 22), 1)

        pygame.draw.rect(screen, (0,180,255), plane_icon)
        draw_text("PLANE", small_font, WHITE, screen, plane_cx, plane_cy + 48)

        tri = [(missile_icon_center[0],   missile_icon_center[1]-30),
               (missile_icon_center[0]-30, missile_icon_center[1]+30),
               (missile_icon_center[0]+30, missile_icon_center[1]+30)]
        pygame.draw.polygon(screen, (220,20,20), tri)
        draw_text("MISSILE", small_font, WHITE, screen, missile_cx, missile_cy + 48)

        hud_surface.fill(PANEL_COLOR)
        hud_clickables.clear()

        display_entities = {}
        for k,v in entities.items():
            display_entities[k] = dict(v)
        if dragging_type and dragging_type in display_entities:
            dx_w,dy_w=drag_world_pos(mouse_pos)
            display_entities[dragging_type]['x']     = dx_w
            display_entities[dragging_type]['y']     = dy_w
            display_entities[dragging_type]['angle'] = current_drag_angle

        cy_surf = 8

        for et, data in display_entities.items():
            real_x = int(data['x']*2)
            real_y = int(-data['y']*2)
            angle  = int(data['angle'])
            name   = "PLANE" if et=='plane' else "MISSILE"
            extra  = data.get('extra', [])

            hdr = small_font.render(f"── {name} ──", True, (200,200,200))
            hud_surface.blit(hdr, (8, cy_surf))
            cy_surf += 24

            fixed_fields = [
                (f"X: {real_x}m",    real_x, 'x_real'),
                (f"Y: {real_y}m",    real_y, 'y_real'),
                (f"Angle: {angle}°", angle,  'angle'),
            ]

            for label, val, fkey in fixed_fields:
                is_active=(active_edit and active_edit['ent']==et and active_edit['field']==fkey)
                screen_rect=pygame.Rect(PANEL_X+8, HUD_TOP+cy_surf-hud_scroll, FIELD_W, ROW_H-2)
                surf_rect  =pygame.Rect(8, cy_surf, FIELD_W, ROW_H-2)

                if is_active:
                    pygame.draw.rect(hud_surface,(255,255,255),surf_rect,border_radius=3)
                    pygame.draw.rect(hud_surface,(0,120,255),  surf_rect,2,border_radius=3)
                    es=small_font.render(active_edit['text']+"|",True,(0,0,200))
                    hud_surface.blit(es,(surf_rect.x+6,surf_rect.y+5))
                else:
                    hover=(screen_rect.collidepoint(mouse_pos) and dragging_type is None
                           and HUD_TOP<=mouse_pos[1]<=HUD_BOT)
                    bg=(180,220,255) if hover else (70,70,70)
                    bc=(80,150,220) if hover else (90,90,90)
                    pygame.draw.rect(hud_surface,bg,surf_rect,border_radius=3)
                    pygame.draw.rect(hud_surface,bc,surf_rect,1,border_radius=3)
                    ts=small_font.render(label,True,(230,230,230))
                    hud_surface.blit(ts,(surf_rect.x+6,surf_rect.y+5))
                    if hover: pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_IBEAM)

                hud_clickables.append((screen_rect, et, fkey, val))
                cy_surf += ROW_H

            for idx, ex in enumerate(extra):
                fkey  = ('extra', idx)
                label = f"{ex['name']}: {ex['value']}"
                val   = ex['value']

                is_active=(active_edit and active_edit['ent']==et and active_edit['field']==fkey)
                screen_rect=pygame.Rect(PANEL_X+8, HUD_TOP+cy_surf-hud_scroll, FIELD_W, ROW_H-2)
                surf_rect  =pygame.Rect(8, cy_surf, FIELD_W, ROW_H-2)

                if is_active:
                    pygame.draw.rect(hud_surface,(255,255,255),surf_rect,border_radius=3)
                    pygame.draw.rect(hud_surface,(0,120,255),  surf_rect,2,border_radius=3)
                    disp = f"{ex['name']}: {active_edit['text']}|"
                    es=small_font.render(disp,True,(0,0,200))
                    hud_surface.blit(es,(surf_rect.x+6,surf_rect.y+5))
                else:
                    hover=(screen_rect.collidepoint(mouse_pos) and dragging_type is None
                           and HUD_TOP<=mouse_pos[1]<=HUD_BOT)
                    bg=(180,220,255) if hover else (70,70,70)
                    bc=(80,150,220) if hover else (90,90,90)
                    pygame.draw.rect(hud_surface,bg,surf_rect,border_radius=3)
                    pygame.draw.rect(hud_surface,bc,surf_rect,1,border_radius=3)
                    ts=small_font.render(label,True,(230,230,230))
                    hud_surface.blit(ts,(surf_rect.x+6,surf_rect.y+5))
                    if hover: pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_IBEAM)

                hud_clickables.append((screen_rect, et, fkey, val))
                cy_surf += ROW_H

            cy_surf += 10 

        hud_content_h = cy_surf
        MAX_SCROLL    = max(0, hud_content_h - HUD_H)

        clip_rect = pygame.Rect(0, hud_scroll, PANEL_W, HUD_H)
        screen.blit(hud_surface, (PANEL_X, HUD_TOP), clip_rect)

        pygame.draw.line(screen,(80,80,80),(PANEL_X,HUD_TOP),  (1000,HUD_TOP),   1)
        pygame.draw.line(screen,(80,80,80),(PANEL_X,HUD_BOT+1),(1000,HUD_BOT+1), 1)

        if MAX_SCROLL > 0:
            sb_h  = max(20, int(HUD_H * HUD_H / hud_content_h))
            sb_y  = HUD_TOP + int(hud_scroll/MAX_SCROLL*(HUD_H-sb_h))
            pygame.draw.rect(screen,(130,130,130),pygame.Rect(1000-6,sb_y,6,sb_h),border_radius=3)

        draw_text("BACK (ESC)",small_font,(150,150,150),screen,PANEL_X+PANEL_W//2,PANEL_H-20)

        if dragging_type:
            draw_rotated_entity(screen,dragging_type,mouse_pos[0],mouse_pos[1],60,current_drag_angle)

        on_interactive = (
            btn_model.collidepoint(mouse_pos) or
            btn_play.collidepoint(mouse_pos) or
            btn_stop.collidepoint(mouse_pos) or
            btn_reset.collidepoint(mouse_pos) or
            any(r.collidepoint(mouse_pos) and HUD_TOP<=mouse_pos[1]<=HUD_BOT for r,*_ in hud_clickables)
        )
        if not on_interactive:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        # ── EVENTLER ──────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if active_edit:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        _apply_edit(entities, active_edit)
                        active_edit = None
                    elif event.key == pygame.K_ESCAPE:
                        active_edit = None
                    elif event.key == pygame.K_BACKSPACE:
                        active_edit['text'] = active_edit['text'][:-1]
                    elif event.unicode in "0123456789-.":
                        active_edit['text'] += event.unicode
                else:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if active_edit:
                        _apply_edit(entities, active_edit)
                        active_edit = None

                    # ── YENİ: PLAY BUTONU VE YAPAY ZEKAYI TETİKLEME ──
                    if btn_play.collidepoint(event.pos):
                        if not is_playing:
                            if not loaded_model_path:
                                print("Lütfen sağ panelden bir yapay zeka modeli (.zip) seçin!")
                            elif 'plane' not in entities or 'missile' not in entities:
                                print("Haritada hem uçak hem de füze olmalıdır!")
                            else:
                                is_playing = True
                                if not initial_entities_state:
                                    initial_entities_state = copy.deepcopy(entities)
                                
                                try:
                                    loaded_model = PPO.load(loaded_model_path)
                                    env = MissileEnv()
                                    env.reset()
                                    
                                    # Pygame'deki koordinatları Gymnasium ortamına aktar
                                    env.plane.x = entities['plane']['x']
                                    env.plane.y = entities['plane']['y']
                                    env.plane.heading = math.radians(-entities['plane']['angle'])
                                    
                                    env.missile.x = entities['missile']['x']
                                    env.missile.y = entities['missile']['y']
                                    env.missile.heading = math.radians(-entities['missile']['angle'])
                                    
                                    # Extra parametreleri aktar
                                    for ex in entities['plane'].get('extra', []):
                                        if ex['name'] == 'Hız': env.plane.speed = ex['value']
                                    for ex in entities['missile'].get('extra', []):
                                        if ex['name'] == 'Motor Gücü': env.missile.thrust = ex['value']
                                    
                                    env.plane.vel_x = env.plane.speed * math.cos(env.plane.heading)
                                    env.plane.vel_y = env.plane.speed * math.sin(env.plane.heading)
                                    env.missile.vel_x = env.missile.speed * math.cos(env.missile.heading)
                                    env.missile.vel_y = env.missile.speed * math.sin(env.missile.heading)
                                    
                                    current_obs = env._get_obs()
                                except Exception as e:
                                    print("Model Yüklenirken Hata Oluştu:", e)
                                    is_playing = False
                                
                    elif btn_stop.collidepoint(event.pos):
                        is_playing = False
                        
                    elif btn_reset.collidepoint(event.pos):
                        is_playing = False
                        if initial_entities_state:
                            entities = copy.deepcopy(initial_entities_state)
                            initial_entities_state = {} 

                    if btn_model.collidepoint(event.pos):
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes('-topmost', True)
                        selected = filedialog.askopenfilename(
                            title="Model Dosyası Seç",
                            filetypes=[("Model dosyaları", "*.zip *.pt *.pth"), ("Tüm dosyalar", "*.*")]
                        )
                        root.destroy()
                        if selected:
                            loaded_model_path = selected
                        continue

                    clicked_hud = False
                    if dragging_type is None and HUD_TOP<=event.pos[1]<=HUD_BOT:
                        for rect, et, fkey, cur_val in hud_clickables:
                            if rect.collidepoint(event.pos) and et in entities:
                                active_edit={'ent':et,'field':fkey,'text':str(cur_val)}
                                clicked_hud=True
                                break

                    if not clicked_hud:
                        if plane_icon.collidepoint(event.pos):
                            dragging_type='plane'; current_drag_angle=0
                        elif missile_icon_rect.collidepoint(event.pos):
                            dragging_type='missile'; current_drag_angle=0
                        elif event.pos[0]<PANEL_X:
                            cwx=(event.pos[0]-camera_x)/zoom
                            cwy=(event.pos[1]-camera_y)/zoom
                            for et,data in list(entities.items()):
                                if abs(data['x']-cwx)<30 and abs(data['y']-cwy)<30:
                                    dragging_type=et
                                    current_drag_angle=data['angle']
                                    break

                elif event.button==3:
                    is_panning=True

                elif event.button in (4,5):
                    if event.pos[0]>=PANEL_X and HUD_TOP<=event.pos[1]<=HUD_BOT:
                        delta=-40 if event.button==4 else 40
                        hud_scroll=max(0,min(MAX_SCROLL,hud_scroll+delta))
                    elif dragging_type:
                        current_drag_angle=(current_drag_angle+(15 if event.button==4 else -15))%360
                    else:
                        mx,my=event.pos
                        if mx<PANEL_X:
                            zf=1.1 if event.button==4 else 1/1.1
                            if 0.2<zoom*zf<3.0:
                                camera_x=mx-(mx-camera_x)*zf
                                camera_y=my-(my-camera_y)*zf
                                zoom*=zf

            if event.type==pygame.MOUSEBUTTONUP:
                if event.button==1 and dragging_type:
                    if mouse_pos[0]<PANEL_X:
                        wx=(mouse_pos[0]-camera_x)/zoom
                        wy=(mouse_pos[1]-camera_y)/zoom
                        if dragging_type not in entities:
                            # Sadece GUI tarafı için varsayılan ekstra değerler
                            default_extra = []
                            if dragging_type == 'plane':
                                default_extra = [{'name':'Hız', 'value':150.0}, {'name':'İrtifa', 'value':5000.0}]
                            elif dragging_type == 'missile':
                                default_extra = [{'name':'Motor Gücü', 'value':500.0}, {'name':'Yakıt', 'value':100.0}]
                            entities[dragging_type]={'x':wx,'y':wy,'angle':current_drag_angle,'extra':default_extra}
                        else:
                            entities[dragging_type]['x']     = wx
                            entities[dragging_type]['y']     = wy
                            entities[dragging_type]['angle'] = current_drag_angle
                    else:
                        if dragging_type in entities:
                            del entities[dragging_type]

                    dragging_type=None
                elif event.button==3:
                    is_panning=False

            if event.type==pygame.MOUSEMOTION and is_panning:
                dx=event.pos[0]-last_mouse_pos[0]; dy=event.pos[1]-last_mouse_pos[1]
                camera_x+=dx; camera_y+=dy

        hud_scroll = max(0, min(MAX_SCROLL, hud_scroll))
        last_mouse_pos=mouse_pos
        pygame.display.update()
        clock.tick(60)


def _apply_edit(entities, active_edit):
    """Aktif inline düzenlemeyi entity'e uygular."""
    try:
        v  = float(active_edit['text'])
        et = active_edit['ent']
        fk = active_edit['field']
        if et not in entities:
            return
        if   fk == 'x_real':  entities[et]['x']     =  v/2.0
        elif fk == 'y_real':  entities[et]['y']     = -v/2.0
        elif fk == 'angle':   entities[et]['angle'] =  v%360
        elif isinstance(fk, tuple) and fk[0]=='extra':
            idx = fk[1]
            if idx < len(entities[et].get('extra',[])):
                entities[et]['extra'][idx]['value'] = v
    except ValueError:
        pass


if __name__ == "__main__":
    main_menu()