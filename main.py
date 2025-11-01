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
            history=500,  # Orta geçmiş
            varThreshold=6,  # Dengeli hassasiyet
            detectShadows=False  # Gölgeleri yok say
        )
        
        # İhlal parametreleri
        self.breach_threshold = 0.15  # %15 hareket = ihlal (daha sıkı)
        self.breach_frames = 0  # Ardışık ihlal frame sayısı
        self.breach_frames_required = 6  # 6 frame = daha kararlı tespit
        self.current_motion_ratio = 0.0  # Anlık hareket oranı
        
        # Profesyonel tespit için
        self.prev_frame_gray = None  # Frame differencing için
        self.min_contour_area = 800  # Minimum nesne boyutu (piksel) - daha büyük
        
        # Pencere ayarları
        self.window_name = 'Restricted Area Breach Detector'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # Resizable pencere
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.fullscreen = False  # Tam ekran modu
        
    def mouse_callback(self, event, x, y, flags, param):
        """Fare ile alan seçimi"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            print(f"[>] Selection started: ({x}, {y})")
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end_point = (x, y)
            w = abs(self.end_point[0] - self.start_point[0])
            h = abs(self.end_point[1] - self.start_point[1])
            print(f"[>] Selection done: {w}x{h} pixels")
            
    def draw_interface(self, frame):
        """Kullanıcı arayüzünü çiz"""
        height, width = frame.shape[:2]
        
        # ALARM durumunda kırmızı ekran ve büyük uyarı
        if self.breach_detected:
            # Yarı saydam kırmızı overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Dinamik font boyutu (pencere boyutuna göre)
            font = cv2.FONT_HERSHEY_SIMPLEX
            base_scale = min(width, height) / 400  # Pencere boyutuna göre ölçeklendir
            
            # Büyük ALARM yazısı
            alarm_text = "!!! ALARM !!!"
            font_scale = 1.8 * base_scale
            thickness = max(3, int(4 * base_scale))
            
            # Metin boyutunu al ve ortala
            (text_width, text_height), _ = cv2.getTextSize(alarm_text, font, font_scale, thickness)
            text_x = (width - text_width) // 2
            text_y = height // 2 - 30
            
            # Siyah gölge
            cv2.putText(frame, alarm_text, (text_x + 3, text_y + 3), 
                       font, font_scale, (0, 0, 0), thickness + 2)
            # Beyaz ana yazı
            cv2.putText(frame, alarm_text, (text_x, text_y), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Alt uyarı
            sub_text = "BREACH DETECTED!"
            font_scale2 = 1.0 * base_scale
            thickness2 = max(2, int(3 * base_scale))
            (text_width2, text_height2), _ = cv2.getTextSize(sub_text, font, font_scale2, thickness2)
            text_x2 = (width - text_width2) // 2
            text_y2 = text_y + int(60 * base_scale)
            
            cv2.putText(frame, sub_text, (text_x2 + 2, text_y2 + 2), 
                       font, font_scale2, (0, 0, 0), thickness2 + 2)
            cv2.putText(frame, sub_text, (text_x2, text_y2), 
                       font, font_scale2, (255, 255, 0), thickness2)
        
        # Korunan alanı göster
        if self.protected_area:
            x, y, w, h = self.protected_area
            color = (0, 0, 255) if self.breach_detected else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Durum metni
            status_text = '[!] BREACH!' if self.breach_detected else '[OK] PROTECTED'
            cv2.putText(frame, status_text, (x, y - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Hareket oranını göster (debug için)
            motion_text = f'Motion: {self.current_motion_ratio*100:.1f}% (Threshold: {self.breach_threshold*100:.1f}%)'
            motion_color = (0, 0, 255) if self.current_motion_ratio > self.breach_threshold else (255, 255, 255)
            cv2.putText(frame, motion_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
        
        # Seçim yapılıyorsa geçici dikdörtgen göster (SARI)
        if self.selecting and self.start_point and self.end_point:
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 255), 3)
            cv2.putText(frame, 'SELECTING...', (self.start_point[0], self.start_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Seçim yapıldı ama henüz kaydedilmedi (TURUNCU)
        elif (not self.selecting and self.start_point and self.end_point 
              and not self.protected_area):
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 165, 255), 3)
            cv2.putText(frame, "Press 's' to save", (self.start_point[0], self.start_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
        # Yardım metni
        help_text = "Mouse:Select | s:Save | r:Reset | f:Fullscreen | q:Quit"
        cv2.putText(frame, help_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
                print(f"[+] Protected area saved: {self.protected_area}")
                return True
        return False
    
    def reset_selection(self):
        """Seçimi sıfırla"""
        was_breached = self.breach_detected
        self.protected_area = None
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.breach_detected = False
        self.breach_frames = 0
        self.prev_frame_gray = None  # Frame diff'i sıfırla
        if was_breached:
            print("[+] Alarm reset - system ready")
        else:
            print("[+] Selection reset")
    
    def toggle_fullscreen(self):
        """Tam ekran modunu aç/kapat"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("[+] Fullscreen ON")
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print("[+] Fullscreen OFF")
    
    def detect_motion_in_area(self, frame):
        """Profesyonel hareket tespiti - Multi-method detection"""
        if not self.protected_area:
            return False
        
        x, y, w, h = self.protected_area
        roi = frame[y:y+h, x:x+w]
        
        # Gray scale dönüşümü
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # METHOD 1: Background Subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.0005)
        roi_mask = fg_mask[y:y+h, x:x+w]
        
        # METHOD 2: Frame Differencing (ardışık frame karşılaştırma)
        frame_diff_detected = False
        if self.prev_frame_gray is not None and self.prev_frame_gray.shape == gray.shape:
            frame_diff = cv2.absdiff(self.prev_frame_gray, gray)
            _, frame_diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            frame_diff_detected = cv2.countNonZero(frame_diff_thresh) > (w * h * 0.05)
        
        self.prev_frame_gray = gray.copy()
        
        # Morfolojik operasyonlar (gürültü temizleme)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Contour detection (gerçek nesneleri bul)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Büyük contourları say
        significant_contours = 0
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                significant_contours += 1
        
        # Hareket oranı hesapla
        motion_pixels = cv2.countNonZero(roi_mask)
        total_pixels = w * h
        motion_ratio = motion_pixels / total_pixels if total_pixels > 0 else 0
        self.current_motion_ratio = motion_ratio
        
        # TRIPLE VERIFICATION: Üç kriterden en az ikisi sağlanmalı
        criteria_met = 0
        
        # Kriter 1: Contour + Motion ratio
        if significant_contours > 0 and motion_ratio > self.breach_threshold:
            criteria_met += 1
        
        # Kriter 2: Frame difference + Motion ratio
        if frame_diff_detected and motion_ratio > self.breach_threshold * 0.8:
            criteria_met += 1
        
        # Kriter 3: Yüksek motion ratio (güçlü hareket)
        if motion_ratio > self.breach_threshold * 1.5:
            criteria_met += 1
        
        # En az 2 kriter sağlanmalı
        motion_detected = criteria_met >= 2
        
        # İhlal kontrolü
        if motion_detected:
            self.breach_frames += 1
            if self.breach_frames >= self.breach_frames_required:
                self.breach_detected = True
                return True
        else:
            # Hareket yoksa sayacı azalt
            if self.breach_frames > 0:
                self.breach_frames = max(0, self.breach_frames - 1)
            
            # Alarm otomatik kapanır
            if self.breach_frames == 0:
                self.breach_detected = False
        
        return self.breach_detected
    
    def run(self):
        """Ana döngü"""
        print("\n" + "="*60)
        print("[*] RESTRICTED AREA BREACH DETECTOR")
        print("="*60)
        print("[>] Starting camera...")
        
        if not self.cap.isOpened():
            print("[!] ERROR: Camera failed to open!")
            return
        
        print("[+] Camera ready!")
        print("\n[?] CONTROLS:")
        print("  Mouse: Select protected area")
        print("  s: Save selection")
        print("  r: Reset selection")
        print("  f: Toggle fullscreen")
        print("  q: Quit")
        print("="*60 + "\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[!] Failed to capture frame!")
                break
            
            # Hareket tespiti yap (korunan alan varsa)
            prev_breach = self.breach_detected
            if self.protected_area:
                breach = self.detect_motion_in_area(frame)
                # Sadece ihlal ilk tespit edildiğinde alarm ver
                if breach and not prev_breach:
                    print(f"[!!!] BREACH DETECTED! [{datetime.now().strftime('%H:%M:%S')}]")
                    print(f"     Motion ratio: {self.current_motion_ratio*100:.1f}%")
            
            # Arayüzü çiz
            display_frame = self.draw_interface(frame.copy())
            
            # Görüntüyü göster
            cv2.imshow(self.window_name, display_frame)
            
            # Klavye kontrolleri
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[x] Shutting down...")
                break
            elif key == ord('s'):
                if self.save_selection():
                    print("[+] Area protected!")
            elif key == ord('r'):
                self.reset_selection()
            elif key == ord('f'):
                self.toggle_fullscreen()
        
        # Temizlik
        self.cap.release()
        cv2.destroyAllWindows()
        print("[+] System closed.\n")

def main():
    detector = RestrictedAreaDetector()
    detector.run()

if __name__ == "__main__":
    main()

