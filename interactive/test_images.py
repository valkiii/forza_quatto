#!/usr/bin/env python3
"""Quick test to verify chip images display correctly."""

import sys
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_chip_images():
    """Test chip images in a simple window."""
    root = tk.Tk()
    root.title("Connect 4 Chip Images Test")
    root.geometry("400x200")
    root.configure(bg='#2c3e50')
    
    # Title
    title_label = tk.Label(
        root,
        text="Connect 4 Chip Images Test",
        font=('Arial', 16, 'bold'),
        bg='#2c3e50',
        fg='white'
    )
    title_label.pack(pady=20)
    
    # Test image loading
    try:
        # Load chip images
        red_img = Image.open('picture/red_chip.png').resize((80, 80), Image.Resampling.LANCZOS)
        yellow_img = Image.open('picture/yellow_chip.png').resize((80, 80), Image.Resampling.LANCZOS)
        
        red_photo = ImageTk.PhotoImage(red_img)
        yellow_photo = ImageTk.PhotoImage(yellow_img)
        
        # Keep references
        red_photo.image = red_img
        yellow_photo.image = yellow_img
        
        # Create frame for chips
        chips_frame = tk.Frame(root, bg='#2c3e50')
        chips_frame.pack(pady=20)
        
        # Red chip
        tk.Label(chips_frame, text="Your Chip", bg='#2c3e50', fg='white', font=('Arial', 12)).grid(row=0, column=0, padx=20)
        red_btn = tk.Button(chips_frame, image=red_photo, width=80, height=80, bg='white')
        red_btn.grid(row=1, column=0, padx=20)
        
        # Yellow chip  
        tk.Label(chips_frame, text="AI Chip", bg='#2c3e50', fg='white', font=('Arial', 12)).grid(row=0, column=1, padx=20)
        yellow_btn = tk.Button(chips_frame, image=yellow_photo, width=80, height=80, bg='white')
        yellow_btn.grid(row=1, column=1, padx=20)
        
        # Success message
        tk.Label(
            root,
            text="✅ Images loaded successfully! These will appear in the game.",
            bg='#2c3e50',
            fg='#27ae60',
            font=('Arial', 10)
        ).pack(pady=10)
        
    except Exception as e:
        # Error message
        tk.Label(
            root,
            text=f"❌ Error loading images: {e}",
            bg='#2c3e50',
            fg='#e74c3c',
            font=('Arial', 10)
        ).pack(pady=10)
    
    # Close button
    tk.Button(
        root,
        text="Close",
        command=root.destroy,
        bg='#3498db',
        fg='white',
        font=('Arial', 12),
        padx=20
    ).pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    test_chip_images()