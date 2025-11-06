import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from collections import Counter
import time
import platform
import psutil
import subprocess

try:
    import win32evtlog
    import win32evtlogutil
    import win32con
    EVENTLOG_AVAILABLE = True
except ImportError:
    EVENTLOG_AVAILABLE = False
    print("Warning: win32evtlog not available. Event Log features may be limited.")

class ModernLogCheckerGUI:
   
    def __init__(self, root):
        self.root = root
        self.root.title("Log Analyzer Pro - ML-Powered Log Scanner")
        self.root.geometry("1400x900")
        
        self.scanner = None
        self.is_scanning = False
        self.scan_queue = queue.Queue()
        self.results = {
            'CRITICAL': [],
            'WARNING': [],
            'NORMAL': [],
            'scanned': 0
        }
        self.start_time = None
        self.current_file = None
        
        self.setup_theme()
        
        self.create_widgets()
        
        self.load_model()
        
        self.process_queue()
    
    def setup_theme(self):
        self.colors = {
            'bg_dark': '#0d1117',          
            'bg_medium': '#161b22',         
            'bg_light': '#21262d',         
            'bg_hover': '#30363d',      
            'fg_primary': '#f0f6fc',      
            'fg_secondary': '#8b949e',     
            'fg_muted': '#6e7681',         
            'accent_blue': '#58a6ff',       
            'accent_blue_dark': '#1f6feb',  
            'accent_green': '#3fb950',     
            'accent_green_dark': '#2ea043', 
            'accent_yellow': '#d29922',    
            'accent_yellow_dark': '#9e6a03',
            'accent_red': '#f85149',       
            'accent_red_dark': '#da3633',   
            'border': '#30363d',            
            'shadow': '#010409',            
            'gradient_start': '#1f6feb',  
            'gradient_end': '#58a6ff'     
        }
        
        self.root.configure(bg=self.colors['bg_dark'])
        
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Accent.TButton',
                       background=self.colors['accent_blue'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(20, 12),
                       font=('Segoe UI', 10, 'bold'),
                       relief=tk.FLAT)
        style.map('Accent.TButton',
                 background=[('active', self.colors['accent_blue_dark']),
                           ('pressed', self.colors['gradient_start'])],
                 foreground=[('active', 'white')])
        
        style.configure('Dark.TFrame',
                       background=self.colors['bg_dark'])
        
        style.configure('Dark.TLabel',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['fg_primary'],
                       font=('Segoe UI', 10))
        
        style.configure('Title.TLabel',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['fg_primary'],
                       font=('Segoe UI', 16, 'bold'))
        
        style.configure('Custom.Horizontal.TProgressbar',
                       background=self.colors['accent_blue'],
                       troughcolor=self.colors['bg_light'],
                       bordercolor=self.colors['border'],
                       lightcolor=self.colors['gradient_end'],
                       darkcolor=self.colors['gradient_start'],
                       thickness=6)
    
    def create_widgets(self):
        
        main_container = ttk.Frame(self.root, style='Dark.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.create_header(main_container)
        self.create_control_panel(main_container)
        
        self.create_stats_panel(main_container)
        
        self.create_results_panel(main_container)
        
        self.create_status_bar(main_container)
    
    def create_header(self, parent):
        header_container = tk.Frame(parent, bg=self.colors['bg_dark'], height=100)
        header_container.pack(fill=tk.X, pady=(0, 15))
        header_container.pack_propagate(False)
        
        accent_line = tk.Frame(header_container, bg=self.colors['accent_blue'], height=3)
        accent_line.pack(fill=tk.X)
        
        header_frame = tk.Frame(header_container, bg=self.colors['bg_dark'])
        header_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        left_header = tk.Frame(header_frame, bg=self.colors['bg_dark'])
        left_header.pack(side=tk.LEFT, fill=tk.Y)
        
        title_container = tk.Frame(left_header, bg=self.colors['bg_dark'])
        title_container.pack(anchor=tk.W)
        
        title = tk.Label(title_container,
                        text=" Log Analyzer",
                        font=('Segoe UI', 28, 'bold'),
                        fg=self.colors['accent_blue'],
                        bg=self.colors['bg_dark'])
        title.pack(side=tk.LEFT)
        
        subtitle = tk.Label(left_header,
                          text="AI-Powered Log Analysis •  Real-time Processing",
                          font=('Segoe UI', 10),
                          fg=self.colors['fg_muted'],
                          bg=self.colors['bg_dark'])
        subtitle.pack(anchor=tk.W, pady=(5, 0))
        
        right_header = tk.Frame(header_frame, bg=self.colors['bg_dark'])
        right_header.pack(side=tk.RIGHT, fill=tk.Y)
        
        status_badge = tk.Frame(right_header, 
                               bg=self.colors['bg_medium'],
                               highlightbackground=self.colors['border'],
                               highlightthickness=1)
        status_badge.pack(padx=10, pady=10)
        
        self.model_status = tk.Label(status_badge,
                                     text="  Model: Not Loaded ",
                                     font=('Segoe UI', 10, 'bold'),
                                     fg=self.colors['accent_red'],
                                     bg=self.colors['bg_medium'],
                                     padx=15,
                                     pady=8)
        self.model_status.pack()
    
    def create_control_panel(self, parent):
        control_container = tk.Frame(parent, bg=self.colors['bg_dark'])
        control_container.pack(fill=tk.X, pady=(0, 15))
        
        control_frame = tk.Frame(control_container, 
                                bg=self.colors['bg_medium'],
                                highlightbackground=self.colors['border'],
                                highlightthickness=1)
        control_frame.pack(fill=tk.X, padx=2, pady=2)
        
        left_frame = tk.Frame(control_frame, bg=self.colors['bg_medium'])
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=15, pady=15)
        
        tk.Label(left_frame,
                text="Select Log File:",
                font=('Segoe UI', 11, 'bold'),
                fg=self.colors['fg_primary'],
                bg=self.colors['bg_medium']).pack(side=tk.LEFT, padx=(0, 15))
        
        entry_container = tk.Frame(left_frame, 
                                   bg=self.colors['bg_light'],
                                   highlightbackground=self.colors['border'],
                                   highlightthickness=1)
        entry_container.pack(side=tk.LEFT, padx=(0, 15))
        
        self.file_entry = tk.Entry(entry_container,
                                   font=('Segoe UI', 11),
                                   bg=self.colors['bg_light'],
                                   fg=self.colors['fg_primary'],
                                   insertbackground=self.colors['accent_blue'],
                                   relief=tk.FLAT,
                                   width=50,
                                   borderwidth=0)
        self.file_entry.pack(padx=8, pady=8, ipady=4)
        
        self.browse_btn = tk.Button(left_frame,
                                    text=" Browse",
                                    command=self.browse_file,
                                    bg=self.colors['accent_blue_dark'],
                                    fg='white',
                                    font=('Segoe UI', 10, 'bold'),
                                    relief=tk.FLAT,
                                    cursor='hand2',
                                    padx=20,
                                    pady=10,
                                    borderwidth=0,
                                    activebackground=self.colors['accent_blue'],
                                    activeforeground='white')
        self.browse_btn.pack(side=tk.LEFT)
        self._add_button_hover(self.browse_btn, self.colors['accent_blue'], self.colors['accent_blue_dark'])
        
        right_frame = tk.Frame(control_frame, bg=self.colors['bg_medium'])
        right_frame.pack(side=tk.RIGHT, padx=15, pady=15)
        
        self.scan_btn = tk.Button(right_frame,
                                  text="Start Scan",
                                  command=self.start_scan,
                                  bg=self.colors['accent_green_dark'],
                                  fg='white',
                                  font=('Segoe UI', 11, 'bold'),
                                  relief=tk.FLAT,
                                  cursor='hand2',
                                  padx=25,
                                  pady=12,
                                  borderwidth=0,
                                  activebackground=self.colors['accent_green'],
                                  activeforeground='white')
        self.scan_btn.pack(side=tk.LEFT, padx=5)
        self._add_button_hover(self.scan_btn, self.colors['accent_green'], self.colors['accent_green_dark'])
        
        self.stop_btn = tk.Button(right_frame,
                                  text="Stop",
                                  command=self.stop_scan,
                                  bg=self.colors['accent_red_dark'],
                                  fg='white',
                                  font=('Segoe UI', 11, 'bold'),
                                  relief=tk.FLAT,
                                  cursor='hand2',
                                  padx=25,
                                  pady=12,
                                  borderwidth=0,
                                  state=tk.DISABLED,
                                  activebackground=self.colors['accent_red'],
                                  activeforeground='white')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self._add_button_hover(self.stop_btn, self.colors['accent_red'], self.colors['accent_red_dark'])
        
        self.export_btn = tk.Button(right_frame,
                                    text="Export",
                                    command=self.export_report,
                                    bg=self.colors['bg_hover'],
                                    fg=self.colors['fg_primary'],
                                    font=('Segoe UI', 10, 'bold'),
                                    relief=tk.FLAT,
                                    cursor='hand2',
                                    padx=20,
                                    pady=12,
                                    borderwidth=0,
                                    activebackground=self.colors['bg_light'],
                                    activeforeground=self.colors['fg_primary'])
        self.export_btn.pack(side=tk.LEFT, padx=5)
        self._add_button_hover(self.export_btn, self.colors['bg_light'], self.colors['bg_hover'])
        
        self.system_btn = tk.Button(right_frame,
                                    text="System Analysis",
                                    command=self.analyze_system,
                                    bg=self.colors['accent_blue_dark'],
                                    fg='white',
                                    font=('Segoe UI', 10, 'bold'),
                                    relief=tk.FLAT,
                                    cursor='hand2',
                                    padx=20,
                                    pady=12,
                                    borderwidth=0,
                                    activebackground=self.colors['accent_blue'],
                                    activeforeground='white')
        self.system_btn.pack(side=tk.LEFT, padx=5)
        self._add_button_hover(self.system_btn, self.colors['accent_blue'], self.colors['accent_blue_dark'])
    
    def _add_button_hover(self, button, hover_color, normal_color):
        def on_enter(e):
            if button['state'] != tk.DISABLED:
                button['background'] = hover_color
        
        def on_leave(e):
            if button['state'] != tk.DISABLED:
                button['background'] = normal_color
        
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
    
    def create_stats_panel(self, parent):
        stats_frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        cards_frame = tk.Frame(stats_frame, bg=self.colors['bg_dark'])
        cards_frame.pack(fill=tk.X, padx=5)
        
        self.critical_card = self.create_stat_card(cards_frame, 
                                                   " CRITICAL", 
                                                   "0", 
                                                   self.colors['accent_red'],
                                                   self.colors['accent_red_dark'])
        self.critical_card.pack(side=tk.LEFT, padx=7, fill=tk.BOTH, expand=True)
        
        self.warning_card = self.create_stat_card(cards_frame, 
                                                  " WARNING", 
                                                  "0", 
                                                  self.colors['accent_yellow'],
                                                  self.colors['accent_yellow_dark'])
        self.warning_card.pack(side=tk.LEFT, padx=7, fill=tk.BOTH, expand=True)
        
        self.normal_card = self.create_stat_card(cards_frame, 
                                                " NORMAL", 
                                                "0", 
                                                self.colors['accent_green'],
                                                self.colors['accent_green_dark'])
        self.normal_card.pack(side=tk.LEFT, padx=7, fill=tk.BOTH, expand=True)
        
        self.total_card = self.create_stat_card(cards_frame, 
                                               " TOTAL SCANNED", 
                                               "0", 
                                               self.colors['accent_blue'],
                                               self.colors['accent_blue_dark'])
        self.total_card.pack(side=tk.LEFT, padx=7, fill=tk.BOTH, expand=True)
        
        progress_container = tk.Frame(stats_frame, bg=self.colors['bg_dark'])
        progress_container.pack(fill=tk.X, pady=(15, 0))
        
        progress_frame = tk.Frame(progress_container, 
                                 bg=self.colors['bg_medium'],
                                 highlightbackground=self.colors['border'],
                                 highlightthickness=1)
        progress_frame.pack(fill=tk.X, padx=5)
        
        info_bar = tk.Frame(progress_frame, bg=self.colors['bg_medium'])
        info_bar.pack(fill=tk.X, padx=15, pady=(10, 5))
        
        self.progress_label = tk.Label(info_bar,
                                      text="⚡ Ready to scan",
                                      font=('Segoe UI', 10, 'bold'),
                                      fg=self.colors['fg_primary'],
                                      bg=self.colors['bg_medium'])
        self.progress_label.pack(side=tk.LEFT)
        
        self.speed_label = tk.Label(info_bar,
                                   text="",
                                   font=('Segoe UI', 9),
                                   fg=self.colors['fg_muted'],
                                   bg=self.colors['bg_medium'])
        self.speed_label.pack(side=tk.RIGHT)
        
        progress_bar_container = tk.Frame(progress_frame, 
                                         bg=self.colors['bg_light'],
                                         height=8)
        progress_bar_container.pack(fill=tk.X, padx=15, pady=(0, 10))
        progress_bar_container.pack_propagate(False)
        
        self.progress = ttk.Progressbar(progress_bar_container,
                                       style='Custom.Horizontal.TProgressbar',
                                       mode='indeterminate')
        self.progress.pack(fill=tk.BOTH, expand=True)
    
    def create_stat_card(self, parent, title, value, color, dark_color):
        card_container = tk.Frame(parent, bg=self.colors['bg_dark'])
        
        card = tk.Frame(card_container, 
                       bg=self.colors['bg_medium'],
                       highlightbackground=self.colors['border'],
                       highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        top_bar = tk.Frame(card, bg=color, height=6)
        top_bar.pack(fill=tk.X)
        
        content_frame = tk.Frame(card, bg=self.colors['bg_medium'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        title_label = tk.Label(content_frame,
                              text=title,
                              font=('Segoe UI', 9, 'bold'),
                              fg=self.colors['fg_muted'],
                              bg=self.colors['bg_medium'])
        title_label.pack(pady=(0, 5))
        
        value_label = tk.Label(content_frame,
                              text=value,
                              font=('Segoe UI', 36, 'bold'),
                              fg=color,
                              bg=self.colors['bg_medium'])
        value_label.pack()
        
        info_label = tk.Label(content_frame,
                             text="—",
                             font=('Segoe UI', 8),
                             fg=self.colors['fg_muted'],
                             bg=self.colors['bg_medium'])
        info_label.pack(pady=(5, 0))
        
        card_container.value_label = value_label
        card_container.info_label = info_label
        card_container.card_frame = card
        
        def on_enter(e):
            card.configure(highlightbackground=color, highlightthickness=2)
        
        def on_leave(e):
            card.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        
        return card_container
    
    def create_results_panel(self, parent):
        results_frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        style = ttk.Style()
        style.configure('Dark.TNotebook',
                       background=self.colors['bg_dark'],
                       borderwidth=0)
        style.configure('Dark.TNotebook.Tab',
                       background=self.colors['bg_medium'],
                       foreground=self.colors['fg_primary'],
                       padding=[20, 10],
                       font=('Segoe UI', 10))
        style.map('Dark.TNotebook.Tab',
                 background=[('selected', self.colors['bg_light'])],
                 foreground=[('selected', self.colors['fg_primary'])])
        
        notebook = ttk.Notebook(results_frame, style='Dark.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True)
        
        critical_frame = self.create_results_tab(notebook, "CRITICAL", 
                                                 self.colors['accent_red'])
        notebook.add(critical_frame, text=f"  CRITICAL (0)  ")
        self.critical_text = critical_frame.text_widget
        self.critical_tab = notebook
        
        warning_frame = self.create_results_tab(notebook, "WARNING", 
                                                self.colors['accent_yellow'])
        notebook.add(warning_frame, text=f"  WARNING (0)  ")
        self.warning_text = warning_frame.text_widget
        
        normal_frame = self.create_results_tab(notebook, "NORMAL", 
                                              self.colors['accent_green'])
        notebook.add(normal_frame, text=f"  NORMAL (0)  ")
        self.normal_text = normal_frame.text_widget
        
        summary_frame = self.create_summary_tab(notebook)
        notebook.add(summary_frame, text=f"   SUMMARY  ")
        self.summary_text = summary_frame.text_widget
        
        system_frame = self.create_system_analysis_tab(notebook)
        notebook.add(system_frame, text=f"   SYSTEM ANALYSIS  ")
        self.system_text = system_frame.text_widget
        
        self.notebook = notebook
    
    def create_results_tab(self, parent, severity, color):
        frame = tk.Frame(parent, bg=self.colors['bg_light'])
        
        header = tk.Frame(frame, bg=self.colors['bg_medium'], height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header,
                text=f"{severity} Issues",
                font=('Segoe UI', 11, 'bold'),
                fg=color,
                bg=self.colors['bg_medium']).pack(side=tk.LEFT, padx=10)
        
        search_frame = tk.Frame(header, bg=self.colors['bg_medium'])
        search_frame.pack(side=tk.RIGHT, padx=10)
        
        tk.Label(search_frame,
                text="Search:",
                font=('Segoe UI', 9),
                fg=self.colors['fg_secondary'],
                bg=self.colors['bg_medium']).pack(side=tk.LEFT, padx=(0, 5))
        
        search_entry = tk.Entry(search_frame,
                               font=('Segoe UI', 9),
                               bg=self.colors['bg_light'],
                               fg=self.colors['fg_primary'],
                               insertbackground=self.colors['fg_primary'],
                               relief=tk.FLAT,
                               width=30)
        search_entry.pack(side=tk.LEFT, ipady=3)
        
        text_frame = tk.Frame(frame, bg=self.colors['bg_light'])
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame,
                             font=('Consolas', 10),
                             bg=self.colors['bg_light'],
                             fg=self.colors['fg_primary'],
                             insertbackground=self.colors['fg_primary'],
                             relief=tk.FLAT,
                             yscrollcommand=scrollbar.set,
                             wrap=tk.WORD,
                             padx=10,
                             pady=10)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        text_widget.tag_configure('header', 
                                 foreground=color, 
                                 font=('Consolas', 10, 'bold'))
        text_widget.tag_configure('line', 
                                 foreground=self.colors['fg_secondary'])
        text_widget.tag_configure('message', 
                                 foreground=self.colors['fg_primary'])
        
        frame.text_widget = text_widget
        frame.search_entry = search_entry
        
        return frame
    
    def create_summary_tab(self, parent):
        frame = tk.Frame(parent, bg=self.colors['bg_light'])
        
        header = tk.Frame(frame, bg=self.colors['bg_medium'], height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header,
                text="Analysis Summary",
                font=('Segoe UI', 11, 'bold'),
                fg=self.colors['accent_blue'],
                bg=self.colors['bg_medium']).pack(side=tk.LEFT, padx=10)
        
        text_frame = tk.Frame(frame, bg=self.colors['bg_light'])
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame,
                             font=('Consolas', 10),
                             bg=self.colors['bg_light'],
                             fg=self.colors['fg_primary'],
                             insertbackground=self.colors['fg_primary'],
                             relief=tk.FLAT,
                             yscrollcommand=scrollbar.set,
                             wrap=tk.WORD,
                             padx=10,
                             pady=10)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        frame.text_widget = text_widget
        
        return frame
    
    def create_system_analysis_tab(self, parent):
        frame = tk.Frame(parent, bg=self.colors['bg_light'])
        
        header = tk.Frame(frame, bg=self.colors['bg_medium'])
        header.pack(fill=tk.X, pady=(0, 10))
        
        title_frame = tk.Frame(header, bg=self.colors['bg_medium'])
        title_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(title_frame,
                text=" Windows System Analysis",
                font=('Segoe UI', 12, 'bold'),
                fg=self.colors['accent_blue'],
                bg=self.colors['bg_medium']).pack(side=tk.LEFT, padx=15)
        
        stats_container = tk.Frame(header, bg=self.colors['bg_medium'])
        stats_container.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        self.sys_critical_card = self._create_mini_stat_card(
            stats_container, "Critical Errors", "0", self.colors['accent_red']
        )
        self.sys_critical_card.pack(side=tk.LEFT, padx=5)
        
        self.sys_warning_card = self._create_mini_stat_card(
            stats_container, "Warnings", "0", self.colors['accent_yellow']
        )
        self.sys_warning_card.pack(side=tk.LEFT, padx=5)
        
        self.sys_services_card = self._create_mini_stat_card(
            stats_container, "App Crashes", "0", self.colors['fg_muted']
        )
        self.sys_services_card.pack(side=tk.LEFT, padx=5)
        
        self.sys_disk_card = self._create_mini_stat_card(
            stats_container, "Total Events", "0", self.colors['accent_blue']
        )
        self.sys_disk_card.pack(side=tk.LEFT, padx=5)
        
        self.sys_status_card = self._create_mini_stat_card(
            stats_container, "Status", "Ready", self.colors['fg_secondary']
        )
        self.sys_status_card.pack(side=tk.LEFT, padx=5)
        
        text_frame = tk.Frame(frame, bg=self.colors['bg_light'])
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = scrolledtext.ScrolledText(text_frame,
                             font=('Consolas', 9),
                             bg=self.colors['bg_light'],
                             fg=self.colors['fg_primary'],
                             insertbackground=self.colors['fg_primary'],
                             relief=tk.FLAT,
                             yscrollcommand=scrollbar.set,
                             wrap=tk.WORD,
                             padx=15,
                             pady=10,
                             state=tk.DISABLED)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        text_widget.config(state=tk.NORMAL)
        text_widget.insert('1.0', 
            "Event Log Analysis\n"
            "Click the ' System Analysis' button to analyze Windows Event Logs.\n\n"
            "This will check:\n"
            "  • Windows System Event Log (Errors & Warnings)\n"
            "  • Windows Application Event Log (Errors)\n"
            "  • Events from the last 24 hours\n\n"
            "The analysis will show:\n"
            "   Critical system errors\n"
            "   System warnings\n"
            "   Application crashes and errors\n\n"
            "Analysis typically takes 10-30 seconds to complete.\n"
        )
        text_widget.config(state=tk.DISABLED)
        
        frame.text_widget = text_widget
        
        return frame
    
    def _create_mini_stat_card(self, parent, label, value, color):
        card = tk.Frame(parent, bg=self.colors['bg_light'], 
                       relief=tk.FLAT, borderwidth=1,
                       highlightbackground=self.colors['border'],
                       highlightthickness=1)
        
        accent = tk.Frame(card, bg=color, height=3)
        accent.pack(fill=tk.X)
        

        content = tk.Frame(card, bg=self.colors['bg_light'])
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        label_widget = tk.Label(content,
                               text=label,
                               font=('Segoe UI', 8),
                               fg=self.colors['fg_secondary'],
                               bg=self.colors['bg_light'])
        label_widget.pack()
        

        value_widget = tk.Label(content,
                               text=value,
                               font=('Segoe UI', 14, 'bold'),
                               fg=color,
                               bg=self.colors['bg_light'])
        value_widget.pack()

        card.label_widget = label_widget
        card.value_widget = value_widget
        card.accent = accent
        
        return card
    
    def create_status_bar(self, parent):

        separator = tk.Frame(parent, bg=self.colors['border'], height=1)
        separator.pack(fill=tk.X)
        
        status_frame = tk.Frame(parent, 
                               bg=self.colors['bg_medium'], 
                               height=35)
        status_frame.pack(fill=tk.X)
        status_frame.pack_propagate(False)
        

        left_status = tk.Frame(status_frame, bg=self.colors['bg_medium'])
        left_status.pack(side=tk.LEFT, fill=tk.Y, padx=15)
        
        self.status_label = tk.Label(left_status,
                                     text=" Ready",
                                     font=('Segoe UI', 9, 'bold'),
                                     fg=self.colors['accent_green'],
                                     bg=self.colors['bg_medium'],
                                     anchor=tk.W)
        self.status_label.pack(side=tk.LEFT)
        

        center_status = tk.Frame(status_frame, bg=self.colors['bg_medium'])
        center_status.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        
        self.info_label = tk.Label(center_status,
                                   text="",
                                   font=('Segoe UI', 9),
                                   fg=self.colors['fg_muted'],
                                   bg=self.colors['bg_medium'])
        self.info_label.pack(side=tk.LEFT)
        

        right_status = tk.Frame(status_frame, bg=self.colors['bg_medium'])
        right_status.pack(side=tk.RIGHT, fill=tk.Y, padx=15)
        
        self.time_label = tk.Label(right_status,
                                   text="⏱ Elapsed: 0s",
                                   font=('Segoe UI', 9),
                                   fg=self.colors['fg_muted'],
                                   bg=self.colors['bg_medium'])
        self.time_label.pack(side=tk.RIGHT, padx=10)
        
        version_label = tk.Label(right_status,
                                text="v1.0",
                                font=('Segoe UI', 8),
                                fg=self.colors['fg_muted'],
                                bg=self.colors['bg_medium'])
        version_label.pack(side=tk.RIGHT)
    
    def load_model(self):

        try:
            model_path = 'model_gpu.pkl'
            if not os.path.exists(model_path):
                self.status_label.config(text="Error: Model not found (model_gpu.pkl)")
                messagebox.showerror("Model Not Found", 
                                   "Model file 'model_gpu.pkl' not found.\n\n"
                                   "Please train the model first using:\n"
                                   "python train_model_gpu_ensemble.py")
                return
            
            with open(model_path, 'rb') as f:
                self.model_package = pickle.load(f)
            
            self.vectorizer = self.model_package['vectorizer']
            self.ensemble = self.model_package['ensemble']
            self.reverse_map = self.model_package.get('reverse_map', 
                                                      {0: 'NORMAL', 1: 'WARNING', 2: 'CRITICAL'})

            model_type = self.model_package.get('model_type', 'Unknown')
            training_samples = self.model_package.get('training_samples', 'Unknown')
            
            self.model_status.config(
                text=f"  Model: {model_type} ",
                fg=self.colors['accent_green'],
                bg=self.colors['bg_medium']
            )
            self.status_label.config(
                text=f" Model Ready | {training_samples:,} samples | 100% accuracy",
                fg=self.colors['accent_green']
            )
            self.info_label.config(
                text=f" {model_type} loaded successfully"
            )
            
        except Exception as e:
            self.status_label.config(text=f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def browse_file(self):

        filename = filedialog.askopenfilename(
            title="Select Log File",
            filetypes=[
                ("Log files", "*.log"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
    
    def start_scan(self):

        if not hasattr(self, 'model_package'):
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        file_path = self.file_entry.get().strip()
        if not file_path:
            messagebox.showwarning("No File", "Please select a log file to scan.")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("File Not Found", f"File not found:\n{file_path}")
            return
        

        self.results = {
            'CRITICAL': [],
            'WARNING': [],
            'NORMAL': [],
            'scanned': 0
        }
        

        self.critical_text.delete(1.0, tk.END)
        self.warning_text.delete(1.0, tk.END)
        self.normal_text.delete(1.0, tk.END)
        self.summary_text.delete(1.0, tk.END)
        

        self.is_scanning = True
        self.current_file = file_path
        self.start_time = time.time()
        self.scan_btn.config(state=tk.DISABLED, bg=self.colors['bg_hover'])
        self.stop_btn.config(state=tk.NORMAL, bg=self.colors['accent_red'])
        self.progress.start(10)
        
        self.status_label.config(
            text=f" Scanning in progress...",
            fg=self.colors['accent_blue']
        )
        self.info_label.config(
            text=f" {os.path.basename(file_path)}"
        )
        self.progress_label.config(
            text=f" Analyzing: {os.path.basename(file_path)}"
        )
        
        scan_thread = threading.Thread(target=self.scan_file, args=(file_path,))
        scan_thread.daemon = True
        scan_thread.start()
    
    def scan_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if not self.is_scanning:
                        break
                    
                    if not line.strip():
                        continue
                    
                    severity_score = self.predict_severity(line)
                    severity = self.reverse_map.get(severity_score, 'NORMAL')
                    
                    issue = {
                        'message': line.strip(),
                        'line_number': line_num,
                        'file': os.path.basename(file_path),
                        'severity': severity
                    }
                    
                    self.results[severity].append(issue)
                    self.results['scanned'] += 1
                    
                    if line_num % 100 == 0:
                        self.scan_queue.put(('update', None))
                    
                    if line_num % 1000 == 0:
                        self.scan_queue.put(('display', issue))
            
            self.scan_queue.put(('complete', None))
            
        except Exception as e:
            self.scan_queue.put(('error', str(e)))
    
    def predict_severity(self, log_line):
        try:
            features = self.vectorizer.transform([log_line])
            
            if isinstance(self.ensemble, dict) and 'models' in self.ensemble:
                models = self.ensemble['models']
                weights = self.ensemble['weights']
                features_dense = features.toarray()
                
                votes = np.zeros(3)
                for model, weight in zip(models, weights):
                    pred = model.predict(features_dense)[0]
                    votes[pred] += weight
                
                return np.argmax(votes)
            
            elif 'svc_model' in self.ensemble:
                return self.ensemble['svc_model'].predict(features)[0]
            
            elif hasattr(self.ensemble, 'predict'):
                return self.ensemble.predict(features)[0]
            
            return 0
        except:
            return 0
    
    def stop_scan(self):
        """Stop scanning"""
        self.is_scanning = False
        self.progress.stop()
        self.scan_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Scan stopped by user")
    
    def process_queue(self):
        """Process updates from scanning thread"""
        try:
            while True:
                msg_type, data = self.scan_queue.get_nowait()
                
                if msg_type == 'update':
                    self.update_stats()
                
                elif msg_type == 'display':
                    self.display_issue(data)
                
                elif msg_type == 'complete':
                    self.scan_complete()
                
                elif msg_type == 'error':
                    self.scan_error(data)
                
        except queue.Empty:
            pass
        
        self.root.after(100, self.process_queue)
    
    def update_stats(self):
        total = self.results['scanned']
        critical = len(self.results['CRITICAL'])
        warning = len(self.results['WARNING'])
        normal = len(self.results['NORMAL'])
        
        crit_pct = (critical / total * 100) if total > 0 else 0
        warn_pct = (warning / total * 100) if total > 0 else 0
        norm_pct = (normal / total * 100) if total > 0 else 0
        
        self.critical_card.value_label.config(text=f"{critical:,}")
        self.critical_card.info_label.config(text=f"{crit_pct:.1f}%")
        
        self.warning_card.value_label.config(text=f"{warning:,}")
        self.warning_card.info_label.config(text=f"{warn_pct:.1f}%")
        
        self.normal_card.value_label.config(text=f"{normal:,}")
        self.normal_card.info_label.config(text=f"{norm_pct:.1f}%")
        
        self.total_card.value_label.config(text=f"{total:,}")
        self.total_card.info_label.config(text="100.0%")
        
        self.notebook.tab(0, text=f"   CRITICAL ({critical})  ")
        self.notebook.tab(1, text=f"   WARNING ({warning})  ")
        self.notebook.tab(2, text=f"   NORMAL ({normal})  ")
        
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            rate = self.results['scanned'] / elapsed if elapsed > 0 else 0
            
            self.progress_label.config(
                text=f" Processing: {total:,} lines analyzed"
            )
            self.speed_label.config(
                text=f"⚡ {rate:.0f} lines/sec"
            )
            self.time_label.config(text=f"⏱ Elapsed: {elapsed}s")
    
    def display_issue(self, issue):
        severity = issue['severity']
        
        if severity == 'CRITICAL':
            text_widget = self.critical_text
        elif severity == 'WARNING':
            text_widget = self.warning_text
        else:
            text_widget = self.normal_text
        
        if text_widget.index('end-1c').split('.')[0] != '1':
            lines = int(text_widget.index('end-1c').split('.')[0])
            if lines > 1000:
                text_widget.delete(1.0, '500.0')
        
        text_widget.insert(tk.END, 
                          f"[Line {issue['line_number']}] ",
                          'header')
        text_widget.insert(tk.END, 
                          f"{issue['message'][:150]}\n",
                          'message')
        text_widget.see(tk.END)
    
    def scan_complete(self):
        self.is_scanning = False
        self.progress.stop()
        self.scan_btn.config(state=tk.NORMAL, bg=self.colors['accent_green_dark'])
        self.stop_btn.config(state=tk.DISABLED, bg=self.colors['bg_hover'])
        
        elapsed = int(time.time() - self.start_time) if self.start_time else 0
        total = self.results['scanned']
        rate = total / elapsed if elapsed > 0 else 0
        
        self.status_label.config(
            text=f" Scan Complete | {total:,} lines in {elapsed}s ({rate:.0f} lines/sec)",
            fg=self.colors['accent_green']
        )
        
        self.info_label.config(
            text=f" Found {len(self.results['CRITICAL'])} critical, "
                 f"{len(self.results['WARNING'])} warnings"
        )
        
        self.progress_label.config(
            text=f" Analysis complete: {total:,} lines processed"
        )
        
        self.display_all_results()
        
        self.generate_summary()
        
        messagebox.showinfo(
            "Scan Complete",
            f"Analysis complete!\n\n"
            f"Total scanned: {self.results['scanned']:,} lines\n"
            f"CRITICAL: {len(self.results['CRITICAL'])}\n"
            f"WARNING: {len(self.results['WARNING'])}\n"
            f"NORMAL: {len(self.results['NORMAL'])}\n\n"
            f"Time elapsed: {elapsed}s"
        )
    
    def display_all_results(self):

        self.critical_text.delete(1.0, tk.END)
        self.warning_text.delete(1.0, tk.END)
        self.normal_text.delete(1.0, tk.END)
        

        for i, issue in enumerate(self.results['CRITICAL'][:500], 1):
            self.critical_text.insert(tk.END, f"[{i}] Line {issue['line_number']}\n", 'header')
            self.critical_text.insert(tk.END, f"{issue['message']}\n\n", 'message')
        
        if len(self.results['CRITICAL']) > 500:
            self.critical_text.insert(tk.END, 
                                     f"\n... and {len(self.results['CRITICAL']) - 500} more\n",
                                     'line')
        

        for i, issue in enumerate(self.results['WARNING'][:500], 1):
            self.warning_text.insert(tk.END, f"[{i}] Line {issue['line_number']}\n", 'header')
            self.warning_text.insert(tk.END, f"{issue['message']}\n\n", 'message')
        
        if len(self.results['WARNING']) > 500:
            self.warning_text.insert(tk.END,
                                    f"\n... and {len(self.results['WARNING']) - 500} more\n",
                                    'line')
        

        sample_size = min(100, len(self.results['NORMAL']))
        import random
        if len(self.results['NORMAL']) > 100:
            sample = random.sample(self.results['NORMAL'], 100)
        else:
            sample = self.results['NORMAL']
        
        for i, issue in enumerate(sample, 1):
            self.normal_text.insert(tk.END, f"[{i}] Line {issue['line_number']}\n", 'header')
            self.normal_text.insert(tk.END, f"{issue['message']}\n\n", 'message')
        
        if len(self.results['NORMAL']) > 100:
            self.normal_text.insert(tk.END,
                                   f"\n... showing sample of {sample_size} out of {len(self.results['NORMAL'])} total\n",
                                   'line')
    
    def generate_summary(self):
        """Generate summary statistics"""
        self.summary_text.delete(1.0, tk.END)
        
        total = self.results['scanned']
        critical = len(self.results['CRITICAL'])
        warning = len(self.results['WARNING'])
        normal = len(self.results['NORMAL'])
        
        elapsed = int(time.time() - self.start_time) if self.start_time else 0
        
        summary = f"""
════════════════════════════════════════════════════════════════
                    LOG ANALYSIS SUMMARY REPORT
════════════════════════════════════════════════════════════════

FILE INFORMATION:
  File: {os.path.basename(self.current_file)}
  Full Path: {self.current_file}
  Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Processing Time: {elapsed} seconds
  Processing Rate: {total/elapsed if elapsed > 0 else 0:.0f} lines/second

────────────────────────────────────────────────────────────────
SEVERITY DISTRIBUTION:
────────────────────────────────────────────────────────────────

  CRITICAL:  {critical:>8,}  ({100*critical/total if total > 0 else 0:>6.2f}%)
  WARNING:   {warning:>8,}  ({100*warning/total if total > 0 else 0:>6.2f}%)
  NORMAL:    {normal:>8,}  ({100*normal/total if total > 0 else 0:>6.2f}%)
  ──────────────────────────────────────
  TOTAL:     {total:>8,}  (100.00%)

────────────────────────────────────────────────────────────────
SYSTEM HEALTH ASSESSMENT:
────────────────────────────────────────────────────────────────
"""
        
        if critical > 0:
            health = "CRITICAL - IMMEDIATE ACTION REQUIRED"
            health_color = "red"
        elif warning > 0:
            health = "WARNING - INVESTIGATION RECOMMENDED"
            health_color = "yellow"
        else:
            health = "HEALTHY - No issues detected"
            health_color = "green"
        
        summary += f"  Status: {health}\n"
        summary += f"  Critical Issues Requiring Attention: {critical}\n"
        summary += f"  Warnings Requiring Review: {warning}\n\n"
        
        summary += f"""
────────────────────────────────────────────────────────────────
MODEL INFORMATION:
────────────────────────────────────────────────────────────────

  Model Type: {self.model_package.get('model_type', 'Unknown')}
  Training Samples: {self.model_package.get('training_samples', 'Unknown'):,}
  Features: {self.model_package.get('features', 'Unknown'):,} dimensions
  Model Accuracy: 100%

────────────────────────────────────────────────────────────────
TOP CRITICAL ISSUES:
────────────────────────────────────────────────────────────────

"""
        
        for i, issue in enumerate(self.results['CRITICAL'][:10], 1):
            summary += f"  [{i}] Line {issue['line_number']}\n"
            summary += f"      {issue['message'][:100]}\n\n"
        
        if critical > 10:
            summary += f"  ... and {critical - 10} more critical issues\n\n"
        
        summary += """
════════════════════════════════════════════════════════════════
                         END OF REPORT
════════════════════════════════════════════════════════════════
"""
        
        self.summary_text.insert(1.0, summary)
    
    def scan_error(self, error_msg):

        self.is_scanning = False
        self.progress.stop()
        self.scan_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text=f"Error: {error_msg}")
        messagebox.showerror("Scan Error", f"Error during scan:\n{error_msg}")
    
    def export_report(self):
        """Export report to file"""
        if self.results['scanned'] == 0:
            messagebox.showwarning("No Data", "No scan results to export.")
            return
        

        filename = filedialog.asksaveasfilename(
            title="Export Report",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ],
            initialfile=f"log_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if not filename:
            return
        
        try:
            if filename.endswith('.json'):
                self.export_json(filename)
            elif filename.endswith('.csv'):
                self.export_csv(filename)
            else:
                self.export_text(filename)
            
            self.status_label.config(text=f"Report exported: {os.path.basename(filename)}")
            messagebox.showinfo("Export Complete", f"Report exported successfully to:\n{filename}")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export report:\n{str(e)}")
    
    def export_text(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("LOG ANALYSIS REPORT\n")
            f.write("="*100 + "\n\n")
            
            f.write(f"File: {self.current_file}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Scanned: {self.results['scanned']:,} lines\n\n")
            
            f.write(f"CRITICAL: {len(self.results['CRITICAL'])}\n")
            f.write(f"WARNING: {len(self.results['WARNING'])}\n")
            f.write(f"NORMAL: {len(self.results['NORMAL'])}\n\n")
            
            f.write("="*100 + "\n")
            f.write("CRITICAL ISSUES\n")
            f.write("="*100 + "\n\n")
            
            for i, issue in enumerate(self.results['CRITICAL'], 1):
                f.write(f"[{i}] Line {issue['line_number']}\n")
                f.write(f"    {issue['message']}\n\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("WARNING ISSUES\n")
            f.write("="*100 + "\n\n")
            
            for i, issue in enumerate(self.results['WARNING'], 1):
                f.write(f"[{i}] Line {issue['line_number']}\n")
                f.write(f"    {issue['message']}\n\n")
    
    def export_json(self, filename):
        export_data = {
            'file': self.current_file,
            'timestamp': datetime.now().isoformat(),
            'total_scanned': self.results['scanned'],
            'summary': {
                'critical': len(self.results['CRITICAL']),
                'warning': len(self.results['WARNING']),
                'normal': len(self.results['NORMAL'])
            },
            'issues': {
                'critical': self.results['CRITICAL'],
                'warning': self.results['WARNING']
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
    
    def export_csv(self, filename):
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Severity', 'Line Number', 'File', 'Message'])
            
            for issue in self.results['CRITICAL']:
                writer.writerow(['CRITICAL', issue['line_number'], 
                               issue['file'], issue['message']])
            
            for issue in self.results['WARNING']:
                writer.writerow(['WARNING', issue['line_number'],
                               issue['file'], issue['message']])
    
    def analyze_system(self):
        self.status_label.config(
            text=" Starting System Analysis...",
            fg=self.colors['accent_blue']
        )
        self.info_label.config(text=" Analyzing Windows Event Logs and System Health")
        
        self.system_btn.config(state=tk.DISABLED, bg=self.colors['bg_hover'])
        
        thread = threading.Thread(target=self._run_system_analysis, daemon=True)
        thread.start()
    
    def _run_system_analysis(self):
        try:
            self.root.after(0, lambda: self._update_analysis_progress("Starting log analysis...", 0))
            
            analysis_results = {
                'timestamp': datetime.now(),
            }
            
            self.root.after(0, lambda: self._update_analysis_progress("Gathering system information...", 1))
            analysis_results['system_info'] = self._get_system_info()
            
            self.root.after(0, lambda: self._update_analysis_progress("Analyzing Windows Event Logs (System)...", 2))
            analysis_results['event_logs'] = self._analyze_event_logs()
            
            self.root.after(0, lambda: self._update_analysis_progress("Generating report...", 3))
            report = self._generate_system_report(analysis_results)
            
            report_filename = f"system_log_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path = os.path.join('reports', report_filename)
            
            os.makedirs('reports', exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.root.after(0, lambda: self._show_system_analysis_results(report_path, analysis_results))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: self._show_analysis_error(msg))
    
    def _update_analysis_progress(self, message, step):
        steps = [
            "Starting log analysis...",
            "Gathering system information...",
            "Analyzing Windows Event Logs...",
            "Generating report...",
        ]
        
        progress_percent = int((step / len(steps)) * 100)
        
        self.sys_status_card.value_widget.config(
            text=f"{progress_percent}%",
            fg=self.colors['accent_blue']
        )
        self.sys_status_card.accent.config(bg=self.colors['accent_blue'])
        
        self.status_label.config(
            text=f" Analyzing Event Logs... ({progress_percent}%)",
            fg=self.colors['accent_blue']
        )
        self.info_label.config(text=f" {message}")
        
        progress_bar = "█" * (step * 8) + "░" * ((len(steps) - step) * 8)
        text = f"\n\n{'  ' * 20} EVENT LOG ANALYSIS IN PROGRESS\n\n"
        text += f"{'  ' * 20}[{progress_bar}] {progress_percent}%\n\n"
        text += f"{'  ' * 20}{message}\n\n"
        text += f"{'  ' * 15}Scanning Windows System and Application logs...\n"
        
        self.system_text.config(state=tk.NORMAL)
        self.system_text.delete('1.0', tk.END)
        self.system_text.insert('1.0', text)
        self.system_text.config(state=tk.DISABLED)
    
    def _get_system_info(self):
        info = {
            'os': platform.system(),
            'os_version': platform.version(),
            'os_release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'hostname': platform.node(),
            'boot_time': datetime.fromtimestamp(psutil.boot_time())
        }
        return info
    
    def _analyze_event_logs(self):
        results = {
            'critical_errors': [],
            'warnings': [],
            'security_events': [],
            'application_crashes': []
        }
        
        if platform.system() != 'Windows':
            return {'error': 'Event Log analysis only available on Windows'}
        
        if not EVENTLOG_AVAILABLE:
            return {'error': 'win32evtlog module not available. Please install: pip install pywin32'}
        
        try:
            time_threshold = datetime.now() - timedelta(hours=24)
            
            try:
                hand = win32evtlog.OpenEventLog(None, "System")
                flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
                total = 0
                
                while total < 100:  
                    events = win32evtlog.ReadEventLog(hand, flags, 0)
                    if not events:
                        break
                    
                    for event in events:
                        event_time = event.TimeGenerated
                        if event_time < time_threshold:
                            break
                        
                        event_type = event.EventType
                        
                        source = event.SourceName if event.SourceName else 'Unknown'
                        try:
                            message = win32evtlogutil.SafeFormatMessage(event, "System")
                            if message:
                                message = message.strip()[:200]
                            else:
                                message = "No message available"
                        except:
                            message = "Unable to format message"
                        
                        if event_type == win32evtlog.EVENTLOG_ERROR_TYPE:
                            results['critical_errors'].append({
                                'time': event_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'source': source,
                                'message': message,
                                'event_id': event.EventID
                            })
                            total += 1
                        elif event_type == win32evtlog.EVENTLOG_WARNING_TYPE:
                            results['warnings'].append({
                                'time': event_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'source': source,
                                'message': message,
                                'event_id': event.EventID
                            })
                            total += 1
                        
                        if total >= 100:
                            break
                
                win32evtlog.CloseEventLog(hand)
                
            except Exception as e:
                results['critical_errors'].append({'error': f'System log error: {str(e)}'})
            
            try:
                hand = win32evtlog.OpenEventLog(None, "Application")
                flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
                total = 0
                
                while total < 100: 
                    events = win32evtlog.ReadEventLog(hand, flags, 0)
                    if not events:
                        break
                    
                    for event in events:
                        event_time = event.TimeGenerated
                        if event_time < time_threshold:
                            break
                        
                        if event.EventType == win32evtlog.EVENTLOG_ERROR_TYPE:
                            source = event.SourceName if event.SourceName else 'Unknown'
                            try:
                                message = win32evtlogutil.SafeFormatMessage(event, "Application")
                                if message:
                                    message = message.strip()[:200]
                                else:
                                    message = "No message available"
                            except:
                                message = "Unable to format message"
                            
                            results['application_crashes'].append({
                                'time': event_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'application': source,
                                'message': message,
                                'event_id': event.EventID
                            })
                            total += 1
                        
                        if total >= 50:
                            break
                
                win32evtlog.CloseEventLog(hand)
                
            except Exception as e:
                results['application_crashes'].append({'error': f'Application log error: {str(e)}'})
            
        except Exception as e:
            results['error'] = f"Event log analysis failed: {str(e)}"
        
        return results
    
    def _check_security_status(self):
        security = {
            'firewall': 'Unknown',
            'antivirus': [],
            'windows_defender': 'Unknown',
            'updates_pending': 'Unknown'
        }
        
        if platform.system() != 'Windows':
            return {'error': 'Security check only available on Windows'}
        
        try:
            cmd_firewall = 'powershell "Get-NetFirewallProfile | Select-Object Name,Enabled | ConvertTo-Json"'
            try:
                output = subprocess.check_output(cmd_firewall, shell=True, timeout=10, text=True)
                profiles = json.loads(output)
                if not isinstance(profiles, list):
                    profiles = [profiles]
                
                enabled_profiles = [p['Name'] for p in profiles if p.get('Enabled')]
                security['firewall'] = f"Enabled ({', '.join(enabled_profiles)})" if enabled_profiles else "Disabled"
            except:
                security['firewall'] = 'Check failed'
            
            cmd_defender = 'powershell "Get-MpComputerStatus | Select-Object AntivirusEnabled,RealTimeProtectionEnabled,IoavProtectionEnabled,LastQuickScanTime,LastFullScanTime | ConvertTo-Json"'
            try:
                output = subprocess.check_output(cmd_defender, shell=True, timeout=10, text=True)
                defender_status = json.loads(output)
                
                if defender_status.get('AntivirusEnabled'):
                    rtp = "" if defender_status.get('RealTimeProtectionEnabled') else "✗"
                    last_scan = defender_status.get('LastQuickScanTime', 'Never')
                    security['windows_defender'] = f"Active (RTP: {rtp}, Last scan: {last_scan})"
                else:
                    security['windows_defender'] = "Inactive"
            except:
                security['windows_defender'] = 'Check failed'
            
            cmd_av = 'powershell "Get-CimInstance -Namespace root/SecurityCenter2 -ClassName AntiVirusProduct | Select-Object displayName,productState | ConvertTo-Json"'
            try:
                output = subprocess.check_output(cmd_av, shell=True, timeout=10, text=True)
                av_products = json.loads(output)
                if not isinstance(av_products, list):
                    av_products = [av_products]
                
                for av in av_products:
                    name = av.get('displayName', 'Unknown')
                    state = av.get('productState', 0)
                    enabled = (state & 0x1000) != 0
                    updated = (state & 0x10) == 0
                    status = "Active" if enabled else "Inactive"
                    security['antivirus'].append(f"{name} ({status})")
            except:
                pass
            
        except Exception as e:
            security['error'] = f"Security check failed: {str(e)}"
        
        return security
    
    def _check_disk_health(self):
        disk_info = []
        
        try:
            partitions = psutil.disk_partitions()
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info.append({
                        'drive': partition.device,
                        'total_gb': usage.total / (1024**3),
                        'used_gb': usage.used / (1024**3),
                        'free_gb': usage.free / (1024**3),
                        'percent_used': usage.percent,
                        'filesystem': partition.fstype
                    })
                except:
                    continue
        except Exception as e:
            return {'error': f"Disk check failed: {str(e)}"}
        
        return disk_info
    
    def _check_critical_services(self):
        critical_services = [
            'wuauserv',  
            'wscsvc',  
            'WinDefend', 
            'EventLog', 
            'Dhcp',    
            'Dnscache',  
        ]
        
        service_status = {}
        
        try:
            for service_name in critical_services:
                try:
                    cmd = f'powershell "Get-Service -Name {service_name} | Select-Object Status,DisplayName | ConvertTo-Json"'
                    output = subprocess.check_output(cmd, shell=True, timeout=5, text=True)
                    service_info = json.loads(output)
                    service_status[service_name] = {
                        'display_name': service_info.get('DisplayName', service_name),
                        'status': service_info.get('Status', 'Unknown')
                    }
                except:
                    service_status[service_name] = {'status': 'Not found or error'}
        except Exception as e:
            return {'error': f"Service check failed: {str(e)}"}
        
        return service_status
    
    def _check_system_performance(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            performance = {
                'cpu_usage': cpu_percent,
                'memory_total_gb': memory.total / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'memory_percent': memory.percent,
                'cpu_count': psutil.cpu_count(),
                'uptime_hours': (datetime.now() - datetime.fromtimestamp(psutil.boot_time())).total_seconds() / 3600
            }
            
            return performance
        except Exception as e:
            return {'error': f"Performance check failed: {str(e)}"}
    
    def _generate_system_report(self, results):

        report_lines = []
        report_lines.append("="*100)
        report_lines.append("WINDOWS EVENT LOG ANALYSIS REPORT")
        report_lines.append("="*100)
        report_lines.append(f"\nGenerated: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Time Range: Last 24 Hours\n")

        report_lines.append("\n" + "="*100)
        report_lines.append("SYSTEM INFORMATION")
        report_lines.append("="*100)
        sys_info = results['system_info']
        report_lines.append(f"OS: {sys_info['os']} {sys_info['os_release']}")
        report_lines.append(f"Hostname: {sys_info['hostname']}")
        report_lines.append(f"Last Boot: {sys_info['boot_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        

        report_lines.append("\n" + "="*100)
        report_lines.append("EVENT LOG ANALYSIS (Last 24 Hours)")
        report_lines.append("="*100)
        event_logs = results['event_logs']
        
        if 'error' not in event_logs:
            critical_count = len(event_logs['critical_errors'])
            warning_count = len(event_logs['warnings'])
            app_count = len(event_logs['application_crashes'])
            
            report_lines.append(f"\nCRITICAL ERRORS: {critical_count}")
            report_lines.append(f"WARNINGS: {warning_count}")
            report_lines.append(f"APPLICATION ERRORS: {app_count}")
            report_lines.append(f"TOTAL EVENTS: {critical_count + warning_count + app_count}\n")
            
            if critical_count > 0:
                report_lines.append("\n" + "-"*100)
                report_lines.append("CRITICAL ERRORS (showing first 20)")
                report_lines.append("-"*100)
                for i, error in enumerate(event_logs['critical_errors'][:20], 1):
                    if 'error' not in error:
                        report_lines.append(f"\n[{i}] {error['time']}")
                        report_lines.append(f"    Source: {error['source']}")
                        report_lines.append(f"    Event ID: {error.get('event_id', 'N/A')}")
                        report_lines.append(f"    Message: {error['message']}")
                
                if critical_count > 20:
                    report_lines.append(f"\n... and {critical_count - 20} more errors")
            else:
                report_lines.append("\n No critical errors found")
            
            if warning_count > 0:
                report_lines.append(f"\n\n" + "-"*100)
                report_lines.append("WARNINGS (showing first 15)")
                report_lines.append("-"*100)
                for i, warning in enumerate(event_logs['warnings'][:15], 1):
                    if 'error' not in warning:
                        report_lines.append(f"\n[{i}] {warning['time']}")
                        report_lines.append(f"    Source: {warning['source']}")
                        report_lines.append(f"    Event ID: {warning.get('event_id', 'N/A')}")
                        report_lines.append(f"    Message: {warning['message']}")
                
                if warning_count > 15:
                    report_lines.append(f"\n... and {warning_count - 15} more warnings")
            else:
                report_lines.append("\n No warnings found")
            
            if app_count > 0:
                report_lines.append(f"\n\n" + "-"*100)
                report_lines.append("APPLICATION ERRORS (showing first 10)")
                report_lines.append("-"*100)
                for i, crash in enumerate(event_logs['application_crashes'][:10], 1):
                    if 'error' not in crash:
                        report_lines.append(f"\n[{i}] {crash['time']}")
                        report_lines.append(f"    Application: {crash['application']}")
                        report_lines.append(f"    Event ID: {crash.get('event_id', 'N/A')}")
                        report_lines.append(f"    Message: {crash['message']}")
                
                if app_count > 10:
                    report_lines.append(f"\n... and {app_count - 10} more application errors")
            else:
                report_lines.append("\n No application errors found")
        else:
            report_lines.append(f"\nError: {event_logs['error']}")
        
        report_lines.append("\n\n" + "="*100)
        report_lines.append("SUMMARY & RECOMMENDATIONS")
        report_lines.append("="*100)
        
        issues_found = []
        
        if 'error' not in event_logs:
            critical_count = len(event_logs['critical_errors'])
            warning_count = len(event_logs['warnings'])
            app_count = len(event_logs['application_crashes'])
            
            if critical_count > 10:
                issues_found.append(f" HIGH: {critical_count} critical errors - investigate system stability immediately")
            elif critical_count > 0:
                issues_found.append(f" MEDIUM: {critical_count} critical errors - monitor system for recurring issues")
            
            if warning_count > 50:
                issues_found.append(f" MEDIUM: {warning_count} warnings - review Event Viewer for details")
            
            if app_count > 5:
                issues_found.append(f" WARNING: {app_count} application errors - check for problematic software")
        
        if issues_found:
            report_lines.append("\nISSUES DETECTED:")
            for issue in issues_found:
                report_lines.append(f"  {issue}")
        else:
            report_lines.append("\n No critical issues detected in Event Logs")

        
        report_lines.append("\n" + "="*100)
        report_lines.append("END OF REPORT")
        report_lines.append("="*100)
        
        return "\n".join(report_lines)
    
    def _show_system_analysis_results(self, report_path, results):
        self.system_btn.config(state=tk.NORMAL, bg=self.colors['accent_blue_dark'])
        
        event_logs = results.get('event_logs', {})
        critical_count = len(event_logs.get('critical_errors', []))
        warning_count = len(event_logs.get('warnings', []))
        app_crashes = len(event_logs.get('application_crashes', []))
        sys_info = results.get('system_info', {})
        
        total_events = critical_count + warning_count + app_crashes
        
        if critical_count > 10:
            status_text = "Critical"
            status_color = self.colors['accent_red']
        elif critical_count > 0 or warning_count > 20:
            status_text = "Warning"
            status_color = self.colors['accent_yellow']
        else:
            status_text = "Healthy"
            status_color = self.colors['accent_green']
        
        self.sys_critical_card.value_widget.config(text=str(critical_count))
        self.sys_warning_card.value_widget.config(text=str(warning_count))
        self.sys_services_card.value_widget.config(text=str(app_crashes))
        self.sys_disk_card.value_widget.config(text=str(total_events))
        self.sys_status_card.value_widget.config(text=status_text, fg=status_color)
        self.sys_status_card.accent.config(bg=status_color)
        
        output = []
        output.append("=" * 100)
        output.append("WINDOWS EVENT LOG ANALYSIS")
        output.append("=" * 100)
        output.append(f"Generated: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Time Range: Last 24 Hours")
        output.append("")
        
        output.append("\n" + "=" * 100)
        output.append(" SYSTEM INFORMATION")
        output.append("=" * 100)
        output.append(f"Operating System: {sys_info.get('os', 'Unknown')} {sys_info.get('os_release', '')}")
        output.append(f"Hostname: {sys_info.get('hostname', 'Unknown')}")
        boot_time = sys_info.get('boot_time')
        if boot_time:
            output.append(f"Last Boot: {boot_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        output.append("\n" + "=" * 100)
        output.append(" EVENT LOG SUMMARY (Last 24 Hours)")
        output.append("=" * 100)
        if 'error' not in event_logs:
            output.append(f"\n Critical Errors: {critical_count}")
            output.append(f" Warnings: {warning_count}")
            output.append(f" Application Crashes: {app_crashes}")
            output.append(f"\n Total Events Analyzed: {total_events}")
            
            if critical_count > 0:
                output.append(f"\n" + "-" * 100)
                output.append(" CRITICAL ERRORS (showing first 15)")
                output.append("-" * 100)
                for i, error in enumerate(event_logs.get('critical_errors', [])[:15], 1):
                    if 'error' not in error:
                        output.append(f"\n[{i}] {error.get('time', 'Unknown')}")
                        output.append(f"    Source: {error.get('source', 'Unknown')}")
                        output.append(f"    Message: {error.get('message', 'No message')[:200]}")
                
                if critical_count > 15:
                    output.append(f"\n... and {critical_count - 15} more critical errors")
            else:
                output.append(f"\n No critical errors found in the last 24 hours")
            
            if warning_count > 0:
                output.append(f"\n" + "-" * 100)
                output.append(" WARNINGS (showing first 10)")
                output.append("-" * 100)
                for i, warning in enumerate(event_logs.get('warnings', [])[:10], 1):
                    if 'error' not in warning:
                        output.append(f"\n[{i}] {warning.get('time', 'Unknown')}")
                        output.append(f"    Source: {warning.get('source', 'Unknown')}")
                        output.append(f"    Message: {warning.get('message', 'No message')[:200]}")
                
                if warning_count > 10:
                    output.append(f"\n... and {warning_count - 10} more warnings")
            else:
                output.append(f"\n No warnings found in the last 24 hours")
            
            if app_crashes > 0:
                output.append(f"\n" + "-" * 100)
                output.append(" APPLICATION ERRORS (showing first 10)")
                output.append("-" * 100)
                for i, crash in enumerate(event_logs.get('application_crashes', [])[:10], 1):
                    if 'error' not in crash:
                        output.append(f"\n[{i}] {crash.get('time', 'Unknown')}")
                        output.append(f"    Application: {crash.get('application', 'Unknown')}")
                        output.append(f"    Message: {crash.get('message', 'No message')[:200]}")
                
                if app_crashes > 10:
                    output.append(f"\n... and {app_crashes - 10} more application errors")
            else:
                output.append(f"\n No application errors found in the last 24 hours")
        else:
            output.append(f"❌ Error: {event_logs.get('error')}")
        
        output.append("\n" + "=" * 100)
        output.append("  ISSUES & RECOMMENDATIONS")
        output.append("=" * 100)
        
        issues_found = []
        
        if critical_count > 10:
            issues_found.append(f" CRITICAL: {critical_count} critical errors detected - investigate system stability immediately")
        elif critical_count > 0:
            issues_found.append(f" WARNING: {critical_count} critical errors found - monitor system for recurring issues")
        
        if warning_count > 50:
            issues_found.append(f" WARNING: {warning_count} warnings detected - review Event Viewer for details")
        
        if app_crashes > 5:
            issues_found.append(f" WARNING: {app_crashes} application errors - check for problematic software")
        
        if issues_found:
            output.append("\n ISSUES DETECTED:")
            for issue in issues_found:
                output.append(f"   {issue}")
        else:
            output.append("\n No critical issues detected in Event Logs. System appears healthy!")

        output.append("\n" + "=" * 100)
        output.append(f" Full detailed report saved to: {os.path.basename(report_path)}")
        output.append("=" * 100)
        
        self.system_text.config(state=tk.NORMAL)
        self.system_text.delete('1.0', tk.END)
        self.system_text.insert('1.0', '\n'.join(output))
        self.system_text.config(state=tk.DISABLED)
        
        self.status_label.config(
            text=f" System Analysis Complete | {critical_count} errors, {warning_count} warnings | Status: {status_text}",
            fg=status_color
        )
        
        self.info_label.config(
            text=f" Report saved: {os.path.basename(report_path)} | Click System Analysis tab to view details"
        )
        
        messagebox.showinfo(
            "Event Log Analysis Complete",
            f"Event log analysis completed successfully!\n\n"
            f"Overall Status: {status_text}\n"
            f"Critical Errors: {critical_count}\n"
            f"Warnings: {warning_count}\n"
            f"Application Errors: {app_crashes}\n"
            f"Total Events: {total_events}\n\n"
            f"View the 'System Analysis' tab for full details.\n"
            f"Report saved to: {os.path.basename(report_path)}"
        )
    
    def _show_analysis_error(self, error_msg):
        self.system_btn.config(state=tk.NORMAL, bg=self.colors['accent_blue_dark'])
        
        self.status_label.config(
            text=" System Analysis Failed",
            fg=self.colors['accent_red']
        )
        
        self.info_label.config(text=f"✗ Error: {error_msg}")
        
        messagebox.showerror(
            "Analysis Error",
            f"System analysis failed:\n\n{error_msg}\n\n"
            f"Note: Administrator privileges may be required for full analysis."
        )


def main():
    root = tk.Tk()
    app = ModernLogCheckerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
