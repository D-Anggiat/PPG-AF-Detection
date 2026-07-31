import serial
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import time

# ===== KONFIGURASI =====
SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 921600
WINDOW_SIZE = 1000

# ===== GLOBAL =====
data_queue = queue.Queue()
norm_buffer = np.zeros(WINDOW_SIZE)
current_stats = {'pred': 'non-AF', 'conf': 0, 'time': 0}
running = True
ser = None

# ============================================================
# SERIAL READER
# ============================================================
def serial_reader():
    global norm_buffer, current_stats, ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        print(f"Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"Error opening serial: {e}")
        return

    buffer = ""
    while running:
        try:
            chunk = ser.read(1024).decode('utf-8', errors='ignore')
        except:
            continue
        if not chunk:
            continue
        buffer += chunk
        lines = buffer.split('\n')
        buffer = lines[-1]
        for line in lines[:-1]:
            line = line.strip()
            if not line:
                continue

            if line.startswith('DATA:'):
                parts = line.split()
                values = []
                for v in parts[1:]:
                    try:
                        values.append(int(v))
                    except:
                        pass
                if len(values) >= WINDOW_SIZE:
                    norm_buffer = np.array(values[:WINDOW_SIZE])
                    data_queue.put('new_data')
            elif line.startswith('PRED:'):
                try:
                    pred = int(line.split()[1])
                    current_stats['pred'] = 'AF' if pred == 1 else 'non-AF'
                    data_queue.put('new_pred')
                except:
                    pass
            elif line.startswith('CONF:'):
                try:
                    current_stats['conf'] = int(line.split()[1])
                except:
                    pass
            elif line.startswith('TIME:'):
                try:
                    current_stats['time'] = int(line.split()[1])
                except:
                    pass

    if ser:
        ser.close()

# ============================================================
# STYLE — palet pastel + rounded card
# ============================================================
COLOR_BG        = '#f3faf9'   # off-white kehijauan, background utama
COLOR_CARD      = '#ffffff'   # kartu putih
COLOR_CARD_ALT  = '#e9f5f3'   # mint muda untuk area di dalam kartu (log box, plot bg)
COLOR_SHADOW    = '#d7e9e6'   # shadow halus di belakang kartu
COLOR_BORDER    = '#e0f0ed'
COLOR_TEXT      = '#25454a'   # teal-slate gelap, kontras baik buat teks
COLOR_SUBTEXT   = '#7fa19c'
COLOR_ACCENT    = '#5fb8ae'   # teal medium
COLOR_ACCENT2   = '#2a8f83'   # teal tua, dipakai untuk ikon & aksen header
COLOR_AF        = '#ffd6d3'   # pastel coral (bg pill alert)
COLOR_AF_TEXT   = '#b1554c'
COLOR_NONAF     = '#d3f0da'   # mint hijau (bg pill normal)
COLOR_NONAF_TEXT= '#3d8a63'
COLOR_WAIT      = '#e3edeb'
COLOR_WAIT_TEXT = '#7c9793'
BTN_BG          = '#daf0ec'   # tombol pastel teal
BTN_HOVER       = '#c3e4de'
STOP_BG         = '#ffdcd9'   # tombol pastel coral
STOP_HOVER      = '#ffc6c1'

def setup_style():
    style = ttk.Style()
    style.theme_use('clam')

    style.configure('Modern.TFrame', background=COLOR_BG)
    style.configure('Card.TFrame', background=COLOR_CARD, relief='flat', borderwidth=0)

    style.configure('Title.TLabel', font=('Segoe UI', 19, 'bold'),
                     background=COLOR_BG, foreground=COLOR_TEXT)
    style.configure('Subtitle.TLabel', font=('Segoe UI', 9),
                     background=COLOR_BG, foreground=COLOR_SUBTEXT)
    style.configure('Header.TLabel', font=('Segoe UI', 9, 'bold'),
                     background=COLOR_CARD, foreground=COLOR_SUBTEXT)
    style.configure('Stat.TLabel', font=('Segoe UI', 10),
                     background=COLOR_CARD, foreground=COLOR_TEXT)

    style.configure('Modern.TButton', font=('Segoe UI', 9, 'bold'), padding=(12, 7),
                     background=BTN_BG, foreground=COLOR_TEXT, borderwidth=0, focusthickness=0)
    style.map('Modern.TButton',
              background=[('active', BTN_HOVER), ('pressed', BTN_HOVER)])

    style.configure('Stop.TButton', font=('Segoe UI', 9, 'bold'), padding=(12, 7),
                     background=STOP_BG, foreground=COLOR_AF_TEXT, borderwidth=0, focusthickness=0)
    style.map('Stop.TButton',
              background=[('active', STOP_HOVER), ('pressed', STOP_HOVER)])

    style.configure('TProgressbar', thickness=14, background=COLOR_ACCENT2,
                     troughcolor=COLOR_WAIT, borderwidth=0, lightcolor=COLOR_ACCENT2,
                     darkcolor=COLOR_ACCENT2)

    style.configure('Modern.TEntry', fieldbackground=COLOR_CARD_ALT, foreground=COLOR_TEXT,
                     insertcolor=COLOR_TEXT, borderwidth=1)

    return style


# ============================================================
# Widget kustom: kartu rounded-corner + shadow halus
# ============================================================
class RoundedCard(tk.Canvas):
    """Canvas yang menggambar rounded-rect + shadow, dan menaruh
    self.body (tk.Frame) di atasnya. Isi widget lain ke self.body."""
    def __init__(self, parent, radius=18, bg_color=COLOR_CARD, shadow=True, **kwargs):
        super().__init__(parent, bg=COLOR_BG, highlightthickness=0, bd=0, **kwargs)
        self.radius = radius
        self.bg_color = bg_color
        self.shadow = shadow
        self.body = tk.Frame(self, bg=bg_color)
        self.body.pack_propagate(False)
        self._win_id = None
        self.bind('<Configure>', self._redraw)

    def _round_pts(self, x1, y1, x2, y2, r):
        r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        return [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
                x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
                x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]

    def _redraw(self, event=None):
        self.delete('shape')
        w, h = self.winfo_width(), self.winfo_height()
        if w < 6 or h < 6:
            return
        sh = 4 if self.shadow else 0
        if self.shadow:
            self.create_polygon(self._round_pts(4, 5 + sh - 4, w - 3, h - 2, self.radius),
                                 smooth=True, fill=COLOR_SHADOW, outline='', tags='shape')
        self.create_polygon(self._round_pts(3, 3, w - 4 - (sh // 2), h - 4 - sh, self.radius),
                             smooth=True, fill=self.bg_color, outline='', tags='shape')
        if self._win_id is None:
            self._win_id = self.create_window(3, 3, window=self.body, anchor='nw', tags='shape_body')
        else:
            self.coords(self._win_id, 3, 3)
        self.body.configure(width=w - 8 - sh, height=h - 8 - sh)
        self.tag_lower('shape')


class RoundedButton(tk.Canvas):
    """Tombol pill/rounded berbasis Canvas — bukan ttk.Button yang selalu kotak."""
    def __init__(self, parent, text, command=None, bg_color=None, hover_color=None,
                 fg_color=None, panel_bg=COLOR_CARD, width=88, height=32, radius=16,
                 font=('Segoe UI', 9, 'bold'), **kwargs):
        super().__init__(parent, width=width, height=height, bg=panel_bg,
                          highlightthickness=0, bd=0, cursor='hand2', **kwargs)
        self.command = command
        self.bg_color = bg_color or BTN_BG
        self.hover_color = hover_color or BTN_HOVER
        self.fg_color = fg_color or COLOR_TEXT
        self.text = text
        self.radius = radius
        self.font = font
        self._current = self.bg_color
        self.bind('<Configure>', self._redraw)
        self.bind('<Enter>', lambda e: self._set_bg(self.hover_color))
        self.bind('<Leave>', lambda e: self._set_bg(self.bg_color))
        self.bind('<ButtonRelease-1>', self._on_click)

    def _set_bg(self, color):
        self._current = color
        self._redraw()

    def _on_click(self, event=None):
        if self.command:
            self.command()

    def _redraw(self, event=None):
        self.delete('all')
        w = self.winfo_width() or int(self['width'])
        h = self.winfo_height() or int(self['height'])
        r = min(self.radius, h / 2)
        pts = [r, 2, w - r, 2, w - 2, 2, w - 2, r, w - 2, h - r, w - 2, h - 2,
               w - r, h - 2, r, h - 2, 2, h - 2, 2, h - r, 2, r, 2, 2]
        self.create_polygon(pts, smooth=True, fill=self._current, outline='')
        self.create_text(w / 2, h / 2, text=self.text, fill=self.fg_color, font=self.font)


class PillLabel(tk.Canvas):
    """Label berbentuk pil (rounded penuh) untuk indikator prediksi."""
    def __init__(self, parent, text="WAIT", bg_color=COLOR_WAIT, fg_color=COLOR_WAIT_TEXT, **kwargs):
        super().__init__(parent, bg=COLOR_CARD, highlightthickness=0, bd=0, height=54, **kwargs)
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.text = text
        self.bind('<Configure>', self._redraw)

    def set_state(self, text, bg_color, fg_color):
        self.text, self.bg_color, self.fg_color = text, bg_color, fg_color
        self._redraw()

    def _redraw(self, event=None):
        self.delete('all')
        w, h = max(self.winfo_width(), 10), max(self.winfo_height(), 10)
        r = h / 2
        pts = [r, 2, w - r, 2, w - 2, 2, w - 2, r,
               w - 2, h - r, w - 2, h - 2, w - r, h - 2, r, h - 2,
               2, h - 2, 2, h - r, 2, r, 2, 2]
        self.create_polygon(pts, smooth=True, fill=self.bg_color, outline='')
        self.create_text(w / 2, h / 2, text=self.text, fill=self.fg_color,
                          font=('Segoe UI', 20, 'bold'))

# ============================================================
# GUI
# ============================================================
class ModernDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("PPG AF Detection Dashboard")
        self.root.geometry("1200x780")
        self.root.configure(bg=COLOR_BG)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.after_update_id = None
        self._is_closing = False
        self._last_pred = None
        self._pulse_state = 0

        main_frame = ttk.Frame(root, style='Modern.TFrame', padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)   # cuma area plot yang melar
        for r in (0, 2, 3):
            main_frame.rowconfigure(r, weight=0)

        # ---------- Header ----------
        header_card = RoundedCard(main_frame, radius=18, shadow=True, height=74)
        header_card.grid(row=0, column=0, sticky='ew', pady=(0, 14))
        header_card.grid_propagate(False)

        icon_holder = tk.Canvas(header_card.body, width=46, height=46, bg=COLOR_CARD, highlightthickness=0)
        icon_holder.pack(side=tk.LEFT, padx=(16, 12), pady=14)
        self._draw_pulse_icon(icon_holder)

        title_box = tk.Frame(header_card.body, bg=COLOR_CARD)
        title_box.pack(side=tk.LEFT, pady=14, fill=tk.Y)
        ttk.Label(title_box, text="PPG AF Detection", style='Title.TLabel',
                  background=COLOR_CARD).pack(anchor='w')
        ttk.Label(title_box, text="Real-time atrial fibrillation monitoring  ·  Golden + Sensor mode",
                  style='Subtitle.TLabel', background=COLOR_CARD).pack(anchor='w')

        status_box = tk.Frame(header_card.body, bg=COLOR_CARD)
        status_box.pack(side=tk.RIGHT, padx=18)
        self.status_dot = tk.Canvas(status_box, width=12, height=12, bg=COLOR_CARD,
                                     highlightthickness=0)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 6))
        self.status_dot_id = self.status_dot.create_oval(1, 1, 11, 11, fill=COLOR_WAIT, outline='')
        self.status_label = ttk.Label(status_box, text="Menunggu data...", style='Subtitle.TLabel',
                                       background=COLOR_CARD)
        self.status_label.pack(side=tk.LEFT)

        # ---------- Plot ----------
        plot_container = RoundedCard(main_frame, radius=20)
        plot_container.grid(row=1, column=0, sticky='nsew', pady=(0, 14))

        self.fig, self.ax = plt.subplots(figsize=(10, 3.2), facecolor=COLOR_CARD)
        self.fig.patch.set_facecolor(COLOR_CARD)
        self.ax.set_facecolor(COLOR_CARD_ALT)
        self.line, = self.ax.plot([], [], color=COLOR_ACCENT2, linewidth=1.8, alpha=0.95)
        self.ax.set_title('PPG Signal', fontsize=12, color=COLOR_TEXT, fontweight='bold')
        self.ax.set_xlabel('Sample', color=COLOR_SUBTEXT)
        self.ax.set_ylabel('Value', color=COLOR_SUBTEXT)
        self.ax.grid(True, linestyle='--', alpha=0.35, color='#c9e6e1')
        self.ax.set_xlim(0, WINDOW_SIZE)
        self.ax.set_ylim(-60, 30)   # rentang awal, nanti di-update otomatis
        self.ax.tick_params(colors=COLOR_SUBTEXT)
        for spine in self.ax.spines.values():
            spine.set_color(COLOR_BORDER)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container.body)
        self.canvas.get_tk_widget().configure(bg=COLOR_CARD, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # ---------- Baris KPI: Prediksi / Confidence / Statistik ----------
        kpi_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        kpi_frame.grid(row=2, column=0, sticky='ew', pady=(0, 14))
        kpi_frame.columnconfigure(0, weight=1)
        kpi_frame.columnconfigure(1, weight=1)
        kpi_frame.columnconfigure(2, weight=1)

        # Prediksi
        pred_card = RoundedCard(kpi_frame, radius=16, height=118)
        pred_card.grid(row=0, column=0, sticky='nsew', padx=(0, 7))
        pred_card.grid_propagate(False)
        self._kpi_header(pred_card.body, "PREDIKSI", COLOR_ACCENT2)
        self.pred_label = PillLabel(pred_card.body, text="WAIT", bg_color=COLOR_WAIT, fg_color=COLOR_WAIT_TEXT)
        self.pred_label.pack(fill=tk.X, padx=16, pady=(4, 16))

        # Confidence
        conf_card = RoundedCard(kpi_frame, radius=16, height=118)
        conf_card.grid(row=0, column=1, sticky='nsew', padx=7)
        conf_card.grid_propagate(False)
        self._kpi_header(conf_card.body, "CONFIDENCE", COLOR_ACCENT)
        self.conf_bar = ttk.Progressbar(conf_card.body, style='TProgressbar', length=150, mode='determinate')
        self.conf_bar.pack(fill=tk.X, padx=16, pady=(6, 6))
        self.conf_text = ttk.Label(conf_card.body, text="0%", style='Stat.TLabel',
                                    font=('Segoe UI', 15, 'bold'), background=COLOR_CARD)
        self.conf_text.pack(pady=(2, 10))

        # Statistik
        stats_card = RoundedCard(kpi_frame, radius=16, height=118)
        stats_card.grid(row=0, column=2, sticky='nsew', padx=(7, 0))
        stats_card.grid_propagate(False)
        self._kpi_header(stats_card.body, "STATISTIK", COLOR_AF)
        self.time_label = ttk.Label(stats_card.body, text="Inference: -- cycles", style='Stat.TLabel',
                                     background=COLOR_CARD)
        self.time_label.pack(anchor='w', padx=16, pady=3)
        self.win_label = ttk.Label(stats_card.body, text="Mode: Sensor", style='Stat.TLabel',
                                    background=COLOR_CARD)
        self.win_label.pack(anchor='w', padx=16, pady=(3, 12))

        # ---------- Log aktivitas + Toolbar kontrol (satu baris, full width) ----------
        log_ctrl_card = RoundedCard(main_frame, radius=16, height=84)
        log_ctrl_card.grid(row=3, column=0, sticky='ew')
        log_ctrl_card.grid_propagate(False)

        row_wrap = tk.Frame(log_ctrl_card.body, bg=COLOR_CARD)
        row_wrap.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # Log (kiri, lebar secukupnya saja)
        left_col = tk.Frame(row_wrap, bg=COLOR_CARD, width=280)
        left_col.pack(side=tk.LEFT, fill=tk.Y)
        left_col.pack_propagate(False)
        header_row = tk.Frame(left_col, bg=COLOR_CARD)
        header_row.pack(fill=tk.X)
        bar = tk.Frame(header_row, bg=COLOR_ACCENT2, width=4, height=13)
        bar.pack(side=tk.LEFT, padx=(0, 7))
        bar.pack_propagate(False)
        ttk.Label(header_row, text="LOG AKTIVITAS", style='Header.TLabel', background=COLOR_CARD).pack(side=tk.LEFT)
        log_wrap = tk.Frame(left_col, bg=COLOR_CARD_ALT, highlightbackground=COLOR_BORDER, highlightthickness=1)
        log_wrap.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.log_text = tk.Text(log_wrap, font=('Consolas', 9), bg=COLOR_CARD_ALT, fg=COLOR_TEXT,
                                 relief='flat', borderwidth=0, padx=8, pady=6, insertbackground=COLOR_TEXT)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Separator tipis
        sep = tk.Frame(row_wrap, bg=COLOR_BORDER, width=1)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=16)

        # Kontrol (kanan, dapat sisa ruang penuh)
        right_col = tk.Frame(row_wrap, bg=COLOR_CARD)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ctrl_frame = tk.Frame(right_col, bg=COLOR_CARD)
        ctrl_frame.place(relx=0, rely=0.5, anchor='w', relwidth=1.0)

        RoundedButton(ctrl_frame, "Sensor", command=self.set_sensor_mode, width=78).pack(side=tk.LEFT, padx=3)
        RoundedButton(ctrl_frame, "Golden", command=self.set_golden_mode, width=78).pack(side=tk.LEFT, padx=3)
        RoundedButton(ctrl_frame, "\u25C0", command=self.prev_golden, width=38).pack(side=tk.LEFT, padx=3)
        RoundedButton(ctrl_frame, "\u25B6", command=self.next_golden, width=38).pack(side=tk.LEFT, padx=3)

        ttk.Label(ctrl_frame, text="Patient ID:", style='Header.TLabel', background=COLOR_CARD).pack(side=tk.LEFT, padx=(16, 6))
        self.patient_entry = ttk.Entry(ctrl_frame, width=8, style='Modern.TEntry')
        self.patient_entry.pack(side=tk.LEFT, padx=3)
        RoundedButton(ctrl_frame, "Load", command=self.load_patient, width=64).pack(side=tk.LEFT, padx=3)

        RoundedButton(ctrl_frame, "Stop", command=self.on_close, width=72,
                      bg_color=STOP_BG, hover_color=STOP_HOVER, fg_color=COLOR_AF_TEXT).pack(side=tk.RIGHT, padx=3)

        self.update_gui()

    # ===== Helper visual: ikon pulse (ECG) tanpa emoji =====
    def _draw_pulse_icon(self, canvas):
        w, h = 46, 46
        r = 10
        pts = [r, 3, w - r, 3, w - 3, 3, w - 3, r,
               w - 3, h - r, w - 3, h - 3, w - r, h - 3, r, h - 3,
               3, h - 3, 3, h - r, 3, r, 3, 3]
        canvas.create_polygon(pts, smooth=True, fill=COLOR_ACCENT2, outline='')
        cy = h / 2
        path = [4, cy, 12, cy, 16, cy - 12, 21, cy + 14, 26, cy - 6, 30, cy, w - 5, cy]
        canvas.create_line(*path, fill='#ffffff', width=2.4, smooth=False,
                            joinstyle='round', capstyle='round')

    # ===== Helper visual: header kecil dengan aksen warna (pengganti ikon emoji) =====
    def _kpi_header(self, parent, text, accent_color):
        row = tk.Frame(parent, bg=COLOR_CARD)
        row.pack(fill=tk.X, padx=16, pady=(12, 4))
        bar = tk.Frame(row, bg=accent_color, width=4, height=13)
        bar.pack(side=tk.LEFT, padx=(0, 7))
        bar.pack_propagate(False)
        ttk.Label(row, text=text, style='Header.TLabel', background=COLOR_CARD).pack(side=tk.LEFT)

    # ===== Kirim perintah ke board =====
    def send_command(self, cmd):
        if ser and ser.is_open:
            ser.write(cmd.encode())
            print(f"Sent: {cmd}")
        else:
            print("Serial port not open")

    def set_sensor_mode(self):
        self.send_command('s')
        self.win_label.config(text="Mode: Sensor")

    def set_golden_mode(self):
        self.send_command('g')
        self.win_label.config(text="Mode: Golden")

    def next_golden(self):
        self.send_command('n')

    def prev_golden(self):
        self.send_command('p')

    def load_patient(self):
        pid = self.patient_entry.get().strip()
        if pid.isdigit():
            self.send_command(f'l{pid}')
        else:
            print("Invalid patient ID")

    # ===== Update GUI =====
    def update_gui(self):
        if not running or self._is_closing:
            return

        new_data = False
        while not data_queue.empty():
            _ = data_queue.get()
            new_data = True

        if new_data and len(norm_buffer) == WINDOW_SIZE:
            self.line.set_data(range(WINDOW_SIZE), norm_buffer)
            # Adjust y-limits dynamically
            ymin = np.min(norm_buffer) - 5
            ymax = np.max(norm_buffer) + 5
            self.ax.set_ylim(ymin, ymax)
            self.ax.relim()
            self.canvas.draw()

            # status dot berkedip halus saat data masuk
            self._pulse_state = 1 - self._pulse_state
            dot_color = COLOR_ACCENT if self._pulse_state else COLOR_ACCENT2
            self.status_dot.itemconfig(self.status_dot_id, fill=dot_color)
            self.status_label.config(text="Menerima data...")

        pred = current_stats['pred']
        if pred == 'AF':
            self.pred_label.set_state(pred, COLOR_AF, COLOR_AF_TEXT)
        else:
            self.pred_label.set_state(pred, COLOR_NONAF, COLOR_NONAF_TEXT)
        self.conf_bar['value'] = current_stats['conf']
        self.conf_text.config(text=f"{current_stats['conf']}%")
        self.time_label.config(text=f"Inference: {current_stats['time']:,} cycles")

        if new_data and pred != self._last_pred:
            self.log_text.config(state='normal')
            timestamp = time.strftime("%H:%M:%S")
            self.log_text.insert('end', f"[{timestamp}] {pred} ({current_stats['conf']}%)\n")
            self.log_text.see('end')
            self.log_text.config(state='disabled')
        self._last_pred = pred

        self.after_update_id = self.root.after(200, self.update_gui)

    def on_close(self):
        global running
        running = False
        self._is_closing = True
        if self.after_update_id is not None:
            try:
                self.root.after_cancel(self.after_update_id)
            except:
                pass
        if ser and ser.is_open:
            ser.close()
        try:
            self.root.quit()
        except:
            pass
        try:
            self.root.destroy()
        except:
            pass

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    thread = threading.Thread(target=serial_reader, daemon=True)
    thread.start()

    root = tk.Tk()
    setup_style()
    app = ModernDashboard(root)
    root.mainloop()
    running = False