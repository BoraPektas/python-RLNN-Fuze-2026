#gui.py görsel kullanıcı arayüzünün çalıştığı yerdir.
import pygame
import sys
import pyperclip
import webbrowser
import math
import subprocess
import os

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

# ─── TRAIN ────────────────────────────────────────────────────────────────────
def train_screen():
    running  = True
    training = False
    process  = None
    log_text = "Eğitim başlatılmamış."
    message  = ""
    msg_timer= 0

    while running:
        screen.fill(BACKGROUND_COLOR)
        mp = pygame.mouse.get_pos()
        pygame.display.set_caption('Train Model')

        draw_text('TRAIN MODEL', caption_font, TEXT_COLOR, screen, 500, 100)
        
        if not training:
            draw_text(log_text, small_font, (150, 150, 150), screen, 500, 250)
            
            btn_start = pygame.Rect(250, 350, 200, 70)
            bc_start = (0, 150, 0) if btn_start.collidepoint(mp) and not training else (0, 100, 0)
            pygame.draw.rect(screen, bc_start, btn_start)
            pygame.draw.rect(screen, TEXT_COLOR, btn_start, 3)
            draw_text("START TRAIN", text_font, WHITE, screen, btn_start.centerx, btn_start.centery)
            
            btn_back = pygame.Rect(550, 350, 200, 70)
            bc_back = (130, 130, 130) if btn_back.collidepoint(mp) else (100, 100, 100)
            pygame.draw.rect(screen, bc_back, btn_back)
            pygame.draw.rect(screen, TEXT_COLOR, btn_back, 3)
            draw_text("BACK", text_font, WHITE, screen, btn_back.centerx, btn_back.centery)
            
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if (btn_start.collidepoint(mp) or btn_back.collidepoint(mp))
                                    else pygame.SYSTEM_CURSOR_ARROW)
        else:
            draw_text("Eğitim devam ediyor...", text_font, (0, 150, 0), screen, 500, 250)
            
            btn_stop = pygame.Rect(350, 350, 300, 70)
            bc_stop = (150, 0, 0) if btn_stop.collidepoint(mp) else (100, 0, 0)
            pygame.draw.rect(screen, bc_stop, btn_stop)
            pygame.draw.rect(screen, TEXT_COLOR, btn_stop, 3)
            draw_text("STOP TRAINING", text_font, WHITE, screen, btn_stop.centerx, btn_stop.centery)
            
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if btn_stop.collidepoint(mp)
                                    else pygame.SYSTEM_CURSOR_ARROW)

        if message:
            draw_text(message, small_font, (50, 200, 50) if "başarı" in message.lower() else (200, 50, 50), screen, 500, 150)
            msg_timer -= 1
            if msg_timer <= 0:
                message = ""

        btn_back = pygame.Rect(700, 600, 200, 60)
        bc_back = (130, 130, 130) if btn_back.collidepoint(mp) else BUTTON_COLOR
        pygame.draw.rect(screen, bc_back, btn_back)
        pygame.draw.rect(screen, TEXT_COLOR, btn_back, 3)
        draw_text("BACK", text_font, WHITE, screen, btn_back.centerx, btn_back.centery)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if process:
                    process.terminate()
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not training and btn_start.collidepoint(mp):
                    src_dir = os.path.dirname(os.path.abspath(__file__))
                    try:
                        process = subprocess.Popen(
                            [sys.executable, os.path.join(src_dir, "train.py"), "--skip-test"],
                            cwd=src_dir
                        )
                        training = True
                        log_text = "Eğitim başladı!"
                        message = "✓ Eğitim başlatıldı"
                        msg_timer = 120
                    except Exception as e:
                        message = f"✗ Hata: {str(e)}"
                        msg_timer = 120
                        
                elif training and btn_stop.collidepoint(mp):
                    if process:
                        process.terminate()
                        process = None
                    training = False
                    log_text = "Eğitim durduruldu."
                    message = "⏹ Eğitim durduruldu"
                    msg_timer = 120
                    
                elif btn_back.collidepoint(mp):
                    if process:
                        process.terminate()
                    running = False
        
        # Eğitim tamamlanmış mı kontrol et
        if training and process and process.poll() is not None:
            training = False
            log_text = "✓ Eğitim tamamlandı!"
            message = "✓ Eğitim başarıyla tamamlandı"
            msg_timer = 180
            process = None

        pygame.display.update()

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


# ─── İSİM SORMA (inline, sağ panelde) ────────────────────────────────────────
def ask_name_inline(prompt_text):
    """
    Ekranın sağ paneli üzerinde küçük bir isim girişi.
    Enter → isim, ESC / boş → None
    """
    input_text = ""
    clock = pygame.time.Clock()
    bx, by, bw, bh = 755, 320, 240, 110
    box  = pygame.Rect(bx, by, bw, bh)
    field= pygame.Rect(bx+10, by+65, bw-20, 30)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return input_text.strip() if input_text.strip() else None
                elif event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    if len(input_text) < 20 and event.unicode.isprintable():
                        input_text += event.unicode

        pygame.draw.rect(screen, (35,35,35),  box, border_radius=6)
        pygame.draw.rect(screen, (100,100,255), box, 2, border_radius=6)

        p_surf = small_font.render(prompt_text, True, (220,220,220))
        screen.blit(p_surf, (bx+10, by+10))
        h_surf = small_font.render("Enter:ok  ESC:iptal", True, (120,120,120))
        screen.blit(h_surf, (bx+10, by+38))

        pygame.draw.rect(screen, (255,255,255), field, border_radius=3)
        pygame.draw.rect(screen, (80,150,255),  field, 2, border_radius=3)
        v_surf = small_font.render(input_text+"|", True, (10,10,10))
        screen.blit(v_surf, (field.x+5, field.y+5))

        pygame.display.update()
        clock.tick(60)


# ─── SIMULATE ─────────────────────────────────────────────────────────────────
def simulate_screen():
    running = True

    camera_x, camera_y = 0.0, 0.0
    zoom = 1.0

    # entities[type] = {'x','y','angle', 'extra': [{'name':str,'value':float}, ...]}
    entities       = {}
    dragging_type  = None
    current_drag_angle = 0
    is_panning     = False
    last_mouse_pos = pygame.mouse.get_pos()
    active_edit    = None   # {'ent','field','text'}   field: 'x_real'|'y_real'|'angle'|('extra',idx)

    PANEL_X    = 750
    PANEL_W    = 250
    PANEL_H    = 750
    PANEL_RECT = pygame.Rect(PANEL_X, 0, PANEL_W, PANEL_H)

    ICON_AREA_H = 430
    ESC_AREA_H  = 40
    HUD_TOP     = ICON_AREA_H
    HUD_BOT     = PANEL_H - ESC_AREA_H
    HUD_H       = HUD_BOT - HUD_TOP

    plane_icon         = pygame.Rect(PANEL_X+95, 100, 60, 60)
    missile_icon_rect   = pygame.Rect(PANEL_X+95, 230, 60, 60)
    missile_icon_center = (PANEL_X+125, 260)

    hud_scroll    = 0
    hud_content_h = 0
    MAX_SCROLL    = 0

    hud_surface = pygame.Surface((PANEL_W, 4000))

    # Tıklanabilir öğeler listesi (her frame doldurulur)
    hud_clickables = []   # (screen_rect, ent_type, field_key, cur_val)
    plus_buttons   = []   # (screen_rect, ent_type) veya (screen_rect, ent_type, 'delete', idx)

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

    ROW_H   = 26
    FIELD_W = PANEL_W - 16

    while running:
        mouse_pos = pygame.mouse.get_pos()

        # ── GRID + ARKA PLAN ──────────────────────────────────────────
        screen.fill(WHITE)

        wl=-camera_x/zoom; wt=-camera_y/zoom
        wr=(PANEL_X-camera_x)/zoom; wb=(PANEL_H-camera_y)/zoom

        for wy in range(int(wt//50)*50, int(wb)+50, 50):
            sy=wy*zoom+camera_y
            pygame.draw.line(screen,
                (120,120,120) if wy%500==0 else (220,220,220),
                (0,sy),(PANEL_X,sy), 3 if wy%500==0 else 1)

        for wx in range(int(wl//50)*50, int(wr)+50, 50):
            sx=wx*zoom+camera_x
            pygame.draw.line(screen,
                (120,120,120) if wx%500==0 else (220,220,220),
                (sx,0),(sx,PANEL_H), 3 if wx%500==0 else 1)

        # ── NESNELER ──────────────────────────────────────────────────
        size=60*zoom
        for et,data in entities.items():
            if et == dragging_type: 
                continue # DÜZELTME 1: Sürüklenen nesnenin haritada iz bırakmaması
                
            sx=data['x']*zoom+camera_x; sy=data['y']*zoom+camera_y
            if sx-size/2>PANEL_X: continue
            draw_rotated_entity(screen,et,sx,sy,size,data['angle'])

        # ── SAĞ PANEL ─────────────────────────────────────────────────
        pygame.draw.rect(screen, PANEL_COLOR, PANEL_RECT)
        pygame.draw.line(screen,(0,0,0),(PANEL_X,0),(PANEL_X,PANEL_H),3)

        draw_text("ENTITIES",small_font,WHITE,screen,PANEL_X+125,50)
        pygame.draw.rect(screen,(0,180,255),plane_icon)
        draw_text("PLANE",small_font,WHITE,screen,PANEL_X+125,180)
        tri=[(missile_icon_center[0],missile_icon_center[1]-30),
             (missile_icon_center[0]-30,missile_icon_center[1]+30),
             (missile_icon_center[0]+30,missile_icon_center[1]+30)]
        pygame.draw.polygon(screen,(220,20,20),tri)
        draw_text("MISSILE",small_font,WHITE,screen,PANEL_X+125,310)

        # ── HUD YÜZEYİ ────────────────────────────────────────────────
        hud_surface.fill(PANEL_COLOR)
        hud_clickables.clear()
        plus_buttons.clear()

        # Sürüklenen entity entities'ten silinmiyor; sadece gösterim için x,y,angle override
        display_entities = {}
        for k,v in entities.items():
            display_entities[k] = dict(v)   # shallow copy
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

            # ── Başlık ──
            hdr = small_font.render(f"── {name} ──", True, (200,200,200))
            hud_surface.blit(hdr, (8, cy_surf))
            cy_surf += 22

            # ── Sabit alanlar ──
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
                    hud_surface.blit(es,(surf_rect.x+6,surf_rect.y+3))
                else:
                    hover=(screen_rect.collidepoint(mouse_pos) and dragging_type is None
                           and HUD_TOP<=mouse_pos[1]<=HUD_BOT)
                    bg=(180,220,255) if hover else (70,70,70)
                    bc=(80,150,220) if hover else (90,90,90)
                    pygame.draw.rect(hud_surface,bg,surf_rect,border_radius=3)
                    pygame.draw.rect(hud_surface,bc,surf_rect,1,border_radius=3)
                    ts=small_font.render(label,True,(230,230,230))
                    hud_surface.blit(ts,(surf_rect.x+6,surf_rect.y+3))
                    if hover: pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_IBEAM)

                hud_clickables.append((screen_rect, et, fkey, val))
                cy_surf += ROW_H

            # ── Ekstra alanlar ──
            for idx, ex in enumerate(extra):
                fkey = ('extra', idx)
                label = f"{ex['name']}: {ex['value']}"
                val   = ex['value']

                DEL_W = 22   # × butonunun genişliği
                is_active=(active_edit and active_edit['ent']==et
                           and active_edit['field']==fkey)

                # Ana alan (× butonu için sağdan yer bırak)
                screen_rect=pygame.Rect(PANEL_X+8, HUD_TOP+cy_surf-hud_scroll, FIELD_W-DEL_W-2, ROW_H-2)
                surf_rect  =pygame.Rect(8, cy_surf, FIELD_W-DEL_W-2, ROW_H-2)

                # × butonu (sağ kenarda)
                del_surf_rect  = pygame.Rect(8+FIELD_W-DEL_W, cy_surf, DEL_W, ROW_H-2)
                del_screen_rect= pygame.Rect(PANEL_X+8+FIELD_W-DEL_W, HUD_TOP+cy_surf-hud_scroll, DEL_W, ROW_H-2)
                del_hover = (del_screen_rect.collidepoint(mouse_pos) and dragging_type is None
                             and HUD_TOP<=mouse_pos[1]<=HUD_BOT)
                del_bg = (200,60,60) if del_hover else (140,40,40)
                pygame.draw.rect(hud_surface, del_bg, del_surf_rect, border_radius=3)
                x_lbl = small_font.render("×", True, (255,200,200))
                hud_surface.blit(x_lbl, (del_surf_rect.centerx - x_lbl.get_width()//2,
                                         del_surf_rect.centery - x_lbl.get_height()//2))
                if del_hover:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)

                if is_active:
                    pygame.draw.rect(hud_surface,(255,255,255),surf_rect,border_radius=3)
                    pygame.draw.rect(hud_surface,(0,120,255),  surf_rect,2,border_radius=3)
                    disp = f"{ex['name']}: {active_edit['text']}|"
                    es=small_font.render(disp,True,(0,0,200))
                    hud_surface.blit(es,(surf_rect.x+6,surf_rect.y+3))
                else:
                    hover=(screen_rect.collidepoint(mouse_pos) and dragging_type is None
                           and HUD_TOP<=mouse_pos[1]<=HUD_BOT)
                    bg=(180,220,255) if hover else (70,70,70)
                    bc=(80,150,220) if hover else (90,90,90)
                    pygame.draw.rect(hud_surface,bg,surf_rect,border_radius=3)
                    pygame.draw.rect(hud_surface,bc,surf_rect,1,border_radius=3)
                    ts=small_font.render(label,True,(230,230,230))
                    hud_surface.blit(ts,(surf_rect.x+6,surf_rect.y+3))
                    if hover: pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_IBEAM)

                hud_clickables.append((screen_rect, et, fkey, val))
                # Silme butonu ayrı listeye
                plus_buttons.append((del_screen_rect, et, 'delete', idx))
                cy_surf += ROW_H

            # ── + Butonu ──
            plus_surf_rect  = pygame.Rect(8, cy_surf, FIELD_W, ROW_H-2)
            plus_screen_rect= pygame.Rect(PANEL_X+8, HUD_TOP+cy_surf-hud_scroll, FIELD_W, ROW_H-2)
            plus_hover = (plus_screen_rect.collidepoint(mouse_pos) and dragging_type is None
                          and HUD_TOP<=mouse_pos[1]<=HUD_BOT)
            plus_bg = (80,160,80) if plus_hover else (55,100,55)
            plus_bc = (120,220,120) if plus_hover else (80,140,80)
            pygame.draw.rect(hud_surface, plus_bg, plus_surf_rect, border_radius=3)
            pygame.draw.rect(hud_surface, plus_bc, plus_surf_rect, 1, border_radius=3)
            plus_label = small_font.render(f"+ Add field to {name}", True, (200,255,200))
            hud_surface.blit(plus_label, (plus_surf_rect.x+6, plus_surf_rect.y+3))
            if plus_hover:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)

            plus_buttons.append((plus_screen_rect, et))
            cy_surf += ROW_H + 10  # entity arası boşluk

        hud_content_h = cy_surf
        MAX_SCROLL    = max(0, hud_content_h - HUD_H)

        clip_rect = pygame.Rect(0, hud_scroll, PANEL_W, HUD_H)
        screen.blit(hud_surface, (PANEL_X, HUD_TOP), clip_rect)

        pygame.draw.line(screen,(80,80,80),(PANEL_X,HUD_TOP),  (1000,HUD_TOP),   1)
        pygame.draw.line(screen,(80,80,80),(PANEL_X,HUD_BOT+1),(1000,HUD_BOT+1), 1)

        # Scroll çubuğu
        if MAX_SCROLL > 0:
            sb_h  = max(20, int(HUD_H * HUD_H / hud_content_h))
            sb_y  = HUD_TOP + int(hud_scroll/MAX_SCROLL*(HUD_H-sb_h))
            pygame.draw.rect(screen,(130,130,130),pygame.Rect(998-6,sb_y,6,sb_h),border_radius=3)

        # ESC etiketi
        draw_text("BACK (ESC)",small_font,(150,150,150),screen,PANEL_X+125,PANEL_H-20)

        # Sürüklenen nesne
        if dragging_type:
            draw_rotated_entity(screen,dragging_type,mouse_pos[0],mouse_pos[1],60,current_drag_angle)

        # İmleç sıfırlama
        on_interactive = (
            any(r.collidepoint(mouse_pos) and HUD_TOP<=mouse_pos[1]<=HUD_BOT for r,*_ in hud_clickables) or
            any(r.collidepoint(mouse_pos) and HUD_TOP<=mouse_pos[1]<=HUD_BOT for r,*_ in plus_buttons)
        )
        if not on_interactive:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        # ── EVENTLER ──────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            # Klavye
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

            # Mouse
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Aktif edit varsa kaydet
                    if active_edit:
                        _apply_edit(entities, active_edit)
                        active_edit = None

                    # + ve x butonlari
                    clicked_plus = False
                    for btn_data in plus_buttons:
                        prect = btn_data[0]
                        if not (prect.collidepoint(event.pos) and HUD_TOP<=event.pos[1]<=HUD_BOT):
                            continue
                        clicked_plus = True
                        if len(btn_data) == 2:
                            pet = btn_data[1]
                            new_name = ask_name_inline("Yeni alan adi:")
                            if new_name:
                                if pet not in entities:
                                    entities[pet]={'x':0,'y':0,'angle':0,'extra':[]}
                                if 'extra' not in entities[pet]:
                                    entities[pet]['extra']=[]
                                entities[pet]['extra'].append({'name':new_name,'value':0.0})
                                hud_scroll = MAX_SCROLL + 999
                        elif len(btn_data) == 4 and btn_data[2] == 'delete':
                            pet = btn_data[1]; idx = btn_data[3]
                            if pet in entities and 'extra' in entities[pet]:
                                if (active_edit and active_edit['ent']==pet
                                        and isinstance(active_edit['field'],tuple)
                                        and active_edit['field']==('extra',idx)):
                                    active_edit = None
                                del entities[pet]['extra'][idx]
                        break

                    if not clicked_plus:
                        # HUD field tıklama
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
                                # DÜZELTME 2: Buradaki oluşturma kodları silindi
                            elif missile_icon_rect.collidepoint(event.pos):
                                dragging_type='missile'; current_drag_angle=0
                                # DÜZELTME 2: Buradaki oluşturma kodları silindi
                            elif event.pos[0]<PANEL_X:
                                cwx=(event.pos[0]-camera_x)/zoom
                                cwy=(event.pos[1]-camera_y)/zoom
                                for et,data in list(entities.items()):
                                    if abs(data['x']-cwx)<30 and abs(data['y']-cwy)<30:
                                        dragging_type=et
                                        current_drag_angle=data['angle']
                                        break  # entities'ten silmiyoruz, extra alanlar korunsun

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
                        # extra alanlar entities'te zaten korunuyor, sadece konum güncelle
                        if dragging_type not in entities:
                            entities[dragging_type]={'x':wx,'y':wy,'angle':current_drag_angle,'extra':[]}
                        else:
                            entities[dragging_type]['x']     = wx
                            entities[dragging_type]['y']     = wy
                            entities[dragging_type]['angle'] = current_drag_angle
                    else:
                        # DÜZELTME 3: Gri alana bırakıldıysa sil
                        if dragging_type in entities:
                            del entities[dragging_type]
                            
                    dragging_type=None
                elif event.button==3:
                    is_panning=False

            if event.type==pygame.MOUSEMOTION and is_panning:
                dx=event.pos[0]-last_mouse_pos[0]; dy=event.pos[1]-last_mouse_pos[1]
                camera_x+=dx; camera_y+=dy

        # Scroll sınırını güncelle (içerik yeni eklendiyse)
        hud_scroll = max(0, min(MAX_SCROLL, hud_scroll))

        last_mouse_pos=mouse_pos
        pygame.display.update()


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