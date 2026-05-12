#gui.py görsel kullanıcı arayüzünün çalıştığı yerdir.
import pygame
import sys
import pyperclip
import webbrowser

pygame.init()

# GEREKLI OLACAK RENKLER
BACKGROUND_COLOR = (180, 180, 180) 
PANEL_COLOR = (50, 50, 50)
BUTTON_COLOR = (120, 120, 120)
TEXT_COLOR = (20, 20, 20)
LINK_COLOR = (70, 130, 180)
WHITE = (255, 255, 255)

# FONTLAR
text_font = pygame.font.SysFont("Consolas", 50, bold = True)
link_font = pygame.font.SysFont("Consolas", 32, bold = True )
caption_font = pygame.font.SysFont("Consolas", 100, bold = True)
small_font = pygame.font.SysFont("Consolas", 20, bold = True)

# EKRAN OLUSTURMA
screen = pygame.display.set_mode((1000,750))

# FONKSIYONLAR 
def draw_text(text, font, color, surface, x, y):
  text_surf = font.render(text, True, color)
  text_rect = text_surf.get_rect(center = (x, y))
  screen.blit(text_surf, text_rect)


def main_menu():
  pygame.display.set_caption('Missile RLNN - Main Menu')
  while(True):

    screen.fill(BACKGROUND_COLOR) 

    mouse_POS = pygame.mouse.get_pos()    

    # MAIN MENU YAZISI
    draw_text("MAIN MENU", caption_font, TEXT_COLOR, screen, 500, 150)
    pygame.draw.line(screen, TEXT_COLOR, (250, 210), (750, 210))

    # BUTONLAR
    btn_w, btn_h = 450, 70
    btn_x = 500 - btn_w // 2

    # Konumları
    btn_simulate = pygame.Rect(btn_x, 250, btn_w, btn_h)
    btn_train = pygame.Rect(btn_x, 350, btn_w, btn_h)
    btn_about = pygame.Rect(btn_x, 450, btn_w, btn_h)
    btn = [btn_simulate, btn_train, btn_about]

    # Çizimleri
    for rect, label in [(btn_simulate, "SIMULATE MODEL"),
                         (btn_train, "TRAIN MODEL"),
                         (btn_about, "ABOUT")]:
      if rect.collidepoint(mouse_POS):

        color = (130, 130, 130)
      else:
        color = BUTTON_COLOR


      pygame.draw.rect(screen, color, rect)
      pygame.draw.rect(screen, TEXT_COLOR, rect, 3)

      draw_text(label, text_font, (255, 255, 255), screen, rect.centerx, rect.centery)

      #İMLEÇ İŞLEMLERİ
      if any(rect.collidepoint(mouse_POS) for rect in btn):
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
      else:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
      

    # TIKLAMA İŞLEMLERİ
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()

      if event.type == pygame.MOUSEBUTTONDOWN:
        if btn_simulate.collidepoint(mouse_POS):
          print("Simülasyon tarafına geçiliyor")
        if btn_train.collidepoint(mouse_POS):
          print("train tarafina geciliyor")
        if btn_about.collidepoint(mouse_POS):
          about_screen()

    
    pygame.display.update()


 
def about_screen():
  running = True
  repo_url = 'https://github.com/BoraPektas/RLNN-Fuze-2026'
  
  link_w, link_h = 800, 40
  link_rect = pygame.Rect(500 - link_w // 2, 500 - link_h // 2, link_w, link_h)

  copy_label_timer = 0


  while(running):

    screen.fill(BACKGROUND_COLOR)
    mouse_POS = pygame.mouse.get_pos()

    pygame.display.set_caption('About')

    # YAZILAR
    draw_text('ABOUT', caption_font, TEXT_COLOR, screen, 500, 100)

    for text, y_value in [('Missile RLNN  Project', 200),
                           ('031890087 Ahmet Şeref Gölcük', 300),
                           ('032490011 Efe Hüner', 350),
                           ('032490028 Bora Pektaş', 400)]:
      draw_text(text, text_font, TEXT_COLOR, screen, 500, y_value)
    
    #LİNK İŞLEMLERİ
    is_hovering = link_rect.collidepoint(mouse_POS)
    if is_hovering:
      current_link_color = (100, 160, 220)
    else:
      current_link_color = LINK_COLOR

    draw_text(repo_url, link_font, current_link_color, screen, 500, 500 )
    

    if is_hovering == False:
      pygame.draw.line(screen, current_link_color, (110, 520), (890, 520), 2)

    if copy_label_timer > 0:
      draw_text('Link coppied to clipboard', small_font, (0, 120, 0), screen, 500, 550)
      copy_label_timer -= 1




    # BACK BUTONU
    btn_back = pygame.Rect(700, 600, 200, 60)
    if btn_back.collidepoint(mouse_POS):
      btn_color = (130, 130, 130)
    else:
      btn_color = BUTTON_COLOR

    pygame.draw.rect(screen, btn_color, btn_back)
    pygame.draw.rect(screen, TEXT_COLOR, btn_back, 3)

    draw_text("BACK", text_font, (255, 255, 255), screen, btn_back.centerx, btn_back.centery )

    #BUTON ÜZERİNDE İMLECİ GETİRME
    if link_rect.collidepoint(mouse_POS) or btn_back.collidepoint(mouse_POS):
      pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
    else:
      pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)



    # EVENTLER
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()

      if event.type == pygame.MOUSEBUTTONDOWN:
        if btn_back.collidepoint(mouse_POS):
          running = False

        if link_rect.collidepoint(mouse_POS):
          pass
          webbrowser.open(repo_url)
          pyperclip.copy(repo_url)
          copy_label_timer = 180

    pygame.display.update()





if __name__ == "__main__":
  main_menu()





    


    
  
    











