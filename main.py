"""
Yasak Bölge İhlali Tespit Sistemi
Restricted Area Breach Detector
Geliştirici: huriyeeym
"""

import cv2
import numpy as np
from datetime import datetime

class RestrictedAreaDetector:
    def __init__(self):
        """Sistem başlatma"""
        self.cap = cv2.VideoCapture(0)
        self.protected_area = None  # Korunan alan koordinatları (x, y, w, h)
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.breach_detected = False
        
        # Hareket tespiti için background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,  # Kaç frame öğrensin (daha hızlı adapte olsun)
            varThreshold=10,  # Hassasiyet (düşük = daha hassas) - 16'dan 10'a düşürüldü
            detectShadows=False  # Gölgeleri yok say (daha net tespit)
        )
        
        # İhlal parametreleri
        self.breach_threshold = 0.15  # %15 hareket = ihlal (daha hassas)
        self.breach_frames = 0  # Ardışık ihlal frame sayısı
        self.breach_frames_required = 5  # 5 frame üst üste = gerçek ihlal (daha hızlı)
        self.current_motion_ratio = 0.0  # Anlık hareket oranı
        
        # Pencere ayarları
        self.window_name = 'Yasak Bolge Ihlali Tespit Sistemi'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Fare ile alan seçimi"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            print(f"[>] Alan secimi basladi: ({x}, {y})")
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end_point = (x, y)
            w = abs(self.end_point[0] - self.start_point[0])
            h = abs(self.end_point[1] - self.start_point[1])
            print(f"[>] Alan secimi tamamlandi: {w}x{h} piksel")
            
    def draw_interface(self, frame):
        """Kullanıcı arayüzünü çiz"""
        height, width = frame.shape[:2]
        
        # ALARM durumunda kırmızı ekran ve büyük uyarı
        if self.breach_detected:
            # Yarı saydam kırmızı overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Büyük ALARM yazısı
            alarm_text = "!!! YASAK BOLGE IHLALI !!!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.5
            thickness = 6
            
            # Metin boyutunu al ve ortala
            (text_width, text_height), _ = cv2.getTextSize(alarm_text, font, font_scale, thickness)
            text_x = (width - text_width) // 2
            text_y = height // 2
            
            # Siyah gölge
            cv2.putText(frame, alarm_text, (text_x + 3, text_y + 3), 
                       font, font_scale, (0, 0, 0), thickness + 2)
            # Beyaz ana yazı
            cv2.putText(frame, alarm_text, (text_x, text_y), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Alt uyarı
            sub_text = "IZINSIZ GIRIS TESPIT EDILDI!"
            font_scale2 = 1.5
            (text_width2, text_height2), _ = cv2.getTextSize(sub_text, font, font_scale2, 4)
            text_x2 = (width - text_width2) // 2
            text_y2 = text_y + 80
            
            cv2.putText(frame, sub_text, (text_x2 + 3, text_y2 + 3), 
                       font, font_scale2, (0, 0, 0), 6)
            cv2.putText(frame, sub_text, (text_x2, text_y2), 
                       font, font_scale2, (255, 255, 0), 4)
        
        # Korunan alanı göster
        if self.protected_area:
            x, y, w, h = self.protected_area
            color = (0, 0, 255) if self.breach_detected else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Durum metni
            status_text = '[!] IHLAL!' if self.breach_detected else '[OK] KORUNUYOR'
            cv2.putText(frame, status_text, (x, y - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Hareket oranını göster (debug için)
            motion_text = f'Hareket: %{self.current_motion_ratio*100:.1f} (Esik: %{self.breach_threshold*100:.0f})'
            motion_color = (0, 0, 255) if self.current_motion_ratio > self.breach_threshold else (255, 255, 255)
            cv2.putText(frame, motion_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
        
        # Seçim yapılıyorsa geçici dikdörtgen göster (SARI)
        if self.selecting and self.start_point and self.end_point:
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 255), 3)
            cv2.putText(frame, 'SECILIYOR...', (self.start_point[0], self.start_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Seçim yapıldı ama henüz kaydedilmedi (TURUNCU)
        elif (not self.selecting and self.start_point and self.end_point 
              and not self.protected_area):
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 165, 255), 3)
            cv2.putText(frame, "'s' tusuna basin!", (self.start_point[0], self.start_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
        # Yardım metni
        cv2.putText(frame, "Fare ile alan secin | 's':Kaydet | 'r':Sifirla | 'q':Cikis", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def save_selection(self):
        """Seçilen alanı kaydet"""
        if self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            if w > 20 and h > 20:  # Minimum alan kontrolü
                self.protected_area = (x, y, w, h)
                print(f"[+] Korunan alan kaydedildi: {self.protected_area}")
                return True
        return False
    
    def reset_selection(self):
        """Seçimi sıfırla"""
        self.protected_area = None
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.breach_detected = False
        self.breach_frames = 0
        print("[+] Secim sifirlandi")
    
    def detect_motion_in_area(self, frame):
        """Korunan alanda hareket tespit et"""
        if not self.protected_area:
            return False
        
        x, y, w, h = self.protected_area
        
        # Background subtraction uygula
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.01)  # Daha hızlı öğrenme
        
        # Hafif gürültü azaltma (daha az filtreleme = daha hassas)
        kernel = np.ones((2, 2), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Sadece korunan alandaki maskeyi al
        roi_mask = fg_mask[y:y+h, x:x+w]
        
        # Hareket eden piksel sayısı
        motion_pixels = cv2.countNonZero(roi_mask)
        total_pixels = w * h
        motion_ratio = motion_pixels / total_pixels if total_pixels > 0 else 0
        
        # Debug: Hareket oranını sakla (ekranda göstermek için)
        self.current_motion_ratio = motion_ratio
        
        # İhlal kontrolü
        if motion_ratio > self.breach_threshold:
            self.breach_frames += 1
            if self.breach_frames >= self.breach_frames_required:
                self.breach_detected = True
                return True
        else:
            # Hareket yoksa sayacı sıfırla
            if self.breach_frames > 0:
                self.breach_frames -= 1
            if self.breach_frames == 0:
                self.breach_detected = False
        
        return self.breach_detected
    
    def run(self):
        """Ana döngü"""
        print("\n" + "="*60)
        print("[*] YASAK BOLGE IHLALI TESPIT SISTEMI")
        print("="*60)
        print("[>] Kamera baslatiliyor...")
        
        if not self.cap.isOpened():
            print("[!] HATA: Kamera acilamadi!")
            return
        
        print("[+] Kamera hazir!")
        print("\n[?] KULLANIM:")
        print("  1. Fare ile korunacak alani secin")
        print("  2. 's' tusuna basarak kaydedin")
        print("  3. 'q' ile cikis yapin")
        print("="*60 + "\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[!] Kamera goruntusu alinamadi!")
                break
            
            # Hareket tespiti yap (korunan alan varsa)
            prev_breach = self.breach_detected
            if self.protected_area:
                breach = self.detect_motion_in_area(frame)
                # Sadece ihlal ilk tespit edildiğinde alarm ver
                if breach and not prev_breach:
                    print(f"[!!!] ALARM! Ihlal tespit edildi! [{datetime.now().strftime('%H:%M:%S')}]")
                    print(f"     Hareket orani: %{self.current_motion_ratio*100:.1f}")
            
            # Arayüzü çiz
            display_frame = self.draw_interface(frame.copy())
            
            # Görüntüyü göster
            cv2.imshow(self.window_name, display_frame)
            
            # Klavye kontrolleri
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[x] Sistem kapatiliyor...")
                break
            elif key == ord('s'):
                if self.save_selection():
                    print("[+] Alan korunmaya basladi!")
            elif key == ord('r'):
                self.reset_selection()
        
        # Temizlik
        self.cap.release()
        cv2.destroyAllWindows()
        print("[+] Sistem kapatildi.\n")

def main():
    detector = RestrictedAreaDetector()
    detector.run()

if __name__ == "__main__":
    main()

