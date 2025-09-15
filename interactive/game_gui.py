"""Interactive Connect 4 GUI for playing against trained Double DQN agent."""

import sys
import os
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np

# Try to import PIL for better images, fallback gracefully
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Connect4Board
from agents.double_dqn_agent import DoubleDQNAgent
from train_fixed_double_dqn import FixedDoubleDQNAgent


class Connect4GUI:
    """Interactive Connect 4 game GUI with clickable board."""
    
    def __init__(self, ai_model_path: str = None):
        self.root = tk.Tk()
        self.root.title("Connect 4 - Human vs AI")
        self.root.geometry("700x800")
        self.root.configure(bg='#2c3e50')
        
        # Game state
        self.board = Connect4Board()
        
        # Randomly assign who goes first
        import random
        if random.random() < 0.5:
            self.human_player = 1  # Human goes first
            self.ai_player = 2     # AI goes second
            self.human_goes_first = True
        else:
            self.human_player = 2  # Human goes second
            self.ai_player = 1     # AI goes first
            self.human_goes_first = False
            
        self.game_over = False
        
        # Colors
        self.colors = {
            'empty': '#ecf0f1',
            'human': '#e74c3c',    # Red for human
            'ai': '#f1c40f',       # Yellow for AI
            'board': '#34495e',
            'hover': '#3498db'
        }
        
        # Initialize AI agent (will be loaded after GUI setup)
        self.ai_agent = None
        self._initial_model_path = ai_model_path
        
        # Load chip images
        self.load_chip_images()
        
        # Initialize GUI first
        self.setup_gui()
        
        # Load AI agent after GUI is ready
        if self._initial_model_path and os.path.exists(self._initial_model_path):
            self.load_ai_agent(self._initial_model_path)
        else:
            self.status_label.config(
                text="❌ No AI model found - Human vs Human mode",
                fg='#e74c3c'
            )
        
        self.update_display()
        
        # If AI goes first, trigger AI move after initial display
        if not self.human_goes_first and self.ai_agent:
            self.root.after(1500, self.ai_move)  # Give time for GUI to load
    
    def load_chip_images(self) -> None:
        """Load the red and yellow chip images or create fallback visuals."""
        self.red_chip_image = None
        self.yellow_chip_image = None
        
        if not HAS_PIL:
            print("⚠️  PIL not available, using enhanced text fallback")
            return
            
        try:
            # Get the path to the picture directory
            current_dir = os.path.dirname(__file__)
            picture_dir = os.path.join(current_dir, "picture")
            
            # Try the new chip images first
            red_path = os.path.join(picture_dir, "red_chip.png")
            yellow_path = os.path.join(picture_dir, "yellow_chip.png")
            
            # Fallback to original names if new ones don't exist
            if not os.path.exists(red_path):
                red_path = os.path.join(picture_dir, "red.png")
            if not os.path.exists(yellow_path):
                yellow_path = os.path.join(picture_dir, "yellow.png")
            
            if os.path.exists(red_path) and os.path.exists(yellow_path):
                try:
                    # Load images with PIL
                    red_image = Image.open(red_path).resize((60, 60), Image.Resampling.LANCZOS)
                    yellow_image = Image.open(yellow_path).resize((60, 60), Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage for tkinter and keep references
                    self.red_chip_image = ImageTk.PhotoImage(red_image)
                    self.yellow_chip_image = ImageTk.PhotoImage(yellow_image)
                    
                    # Keep references to prevent garbage collection
                    self.red_chip_image.image = red_image  
                    self.yellow_chip_image.image = yellow_image
                    
                    print("✅ Chip images loaded successfully")
                    return
                except Exception as e:
                    print(f"⚠️  Error loading images with PIL: {e}")
                    
            # Try tkinter native approach  
            try:
                self.red_chip_image = tk.PhotoImage(file=red_path)
                self.yellow_chip_image = tk.PhotoImage(file=yellow_path)
                
                # Keep references to prevent garbage collection
                self.red_chip_image.image = self.red_chip_image
                self.yellow_chip_image.image = self.yellow_chip_image
                
                print("✅ Chip images loaded with tkinter PhotoImage")
                return
            except Exception as e:
                print(f"⚠️  Error loading images with tkinter: {e}")
                
        except Exception as e:
            print(f"⚠️  General error loading images: {e}")
            
        print("Using enhanced visual fallback")
        
    def load_ai_agent(self, model_path: str) -> None:
        """Load the trained Double DQN agent with correct configuration."""
        try:
            # Use FixedDoubleDQNAgent with exact training configuration
            self.ai_agent = FixedDoubleDQNAgent(
                player_id=self.ai_player,
                state_size=84,  # 2 channels * 6 rows * 7 cols
                action_size=7,
                seed=42,
                # CRITICAL: Match exact training configuration
                gradient_clip_norm=1.0,
                use_huber_loss=True,
                huber_delta=1.0,
                state_normalization=True  # This was missing!
            )
            self.ai_agent.load(model_path, keep_player_id=False)  # Preserve our player_id=2
            self.ai_agent.epsilon = 0.1  # No exploration during gameplay
            self.status_label.config(
                text=f"✅ AI Agent loaded from {os.path.basename(model_path)}",
                fg='#27ae60'
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load AI model: {e}")
            self.ai_agent = None
            self.status_label.config(
                text="❌ No AI model loaded - Human vs Human mode",
                fg='#e74c3c'
            )
    
    def setup_gui(self) -> None:
        """Setup the GUI components."""
        # Title
        title_label = tk.Label(
            self.root,
            text="Connect 4: Human vs AI",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title_label.pack(pady=20)
        
        # Status frame
        status_frame = tk.Frame(self.root, bg='#2c3e50')
        status_frame.pack(pady=10)
        
        self.status_label = tk.Label(
            status_frame,
            text="Loading...",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#95a5a6'
        )
        self.status_label.pack()
        
        # Turn indicator
        initial_text = "Your turn! Click a column to drop your piece." if self.human_goes_first else "AI goes first! Get ready..."
        initial_color = '#e74c3c' if self.human_goes_first else '#f1c40f'
        
        self.turn_label = tk.Label(
            status_frame,
            text=initial_text,
            font=('Arial', 14, 'bold'),
            bg='#2c3e50',
            fg=initial_color
        )
        self.turn_label.pack(pady=5)
        
        # Game board frame with fixed size
        self.board_frame = tk.Frame(self.root, bg='#34495e', padx=10, pady=10)
        self.board_frame.pack(pady=20)
        
        # Configure grid to maintain consistent column widths
        for col in range(7):
            self.board_frame.grid_columnconfigure(col, weight=1, uniform="columns")
        for row in range(6):
            self.board_frame.grid_rowconfigure(row, weight=1, uniform="rows")
        
        # Create board buttons
        self.buttons = []
        for row in range(6):  # Connect 4 is 6x7
            button_row = []
            for col in range(7):
                btn = tk.Button(
                    self.board_frame,
                    text='',
                    width=8,   # Slightly wider for better spacing
                    height=4,  # Slightly taller for better proportion
                    font=('Arial', 14, 'bold'),  # Slightly smaller font to fit better
                    bg=self.colors['empty'],
                    command=lambda c=col: self.human_move(c),
                    relief='raised',
                    bd=3,
                    compound=tk.CENTER,  # Center image and text
                    anchor=tk.CENTER     # Ensure content is centered
                )
                btn.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
                button_row.append(btn)
            self.buttons.append(button_row)
        
        # Control buttons frame
        control_frame = tk.Frame(self.root, bg='#2c3e50')
        control_frame.pack(pady=20)
        
        # New game button
        tk.Button(
            control_frame,
            text="New Game",
            command=self.new_game,
            font=('Arial', 14),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=10)
        
        # Load model button
        tk.Button(
            control_frame,
            text="Load AI Model",
            command=self.load_model_dialog,
            font=('Arial', 14),
            bg='#9b59b6',
            fg='white',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=10)
        
        # Game info frame
        info_frame = tk.Frame(self.root, bg='#2c3e50')
        info_frame.pack(pady=10)
        
        tk.Label(
            info_frame,
            text="Red Buttons = You  |  Yellow Buttons = AI",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#95a5a6'
        ).pack()
        
    def update_display(self) -> None:
        """Update the visual board display."""
        board_state = self.board.get_state()
        
        for row in range(6):
            for col in range(7):
                btn = self.buttons[row][col]
                cell_value = board_state[row, col]
                
                if cell_value == 0:  # Empty
                    btn.config(
                        text='',
                        image='',  # Clear any image
                        font=('Arial', 14, 'bold'),  # Consistent font size
                        bg=self.colors['empty'],
                        fg='black',
                        relief='raised',
                        bd=3,
                        state='normal',
                        width=8,  # Maintain consistent width
                        height=4  # Maintain consistent height
                    )
                elif cell_value == 1:  # Human (Red)
                    if self.red_chip_image:
                        btn.config(
                            text='',
                            image=self.red_chip_image,
                            bg=self.colors['empty'],  # Keep background neutral
                            state='disabled',
                            width=8,  # Maintain consistent width
                            height=4  # Maintain consistent height
                        )
                    else:
                        # Enhanced fallback for human chips
                        btn.config(
                            text='●',  # Solid circle
                            image='',
                            font=('Arial', 18, 'bold'),  # Slightly smaller for consistency
                            bg='#e74c3c',  # Red background
                            fg='white',     # White text for contrast
                            relief='raised',
                            bd=2,
                            activebackground='#c0392b',
                            state='disabled',
                            width=8,  # Maintain consistent width
                            height=4  # Maintain consistent height
                        )
                elif cell_value == 2:  # AI (Yellow)
                    if self.yellow_chip_image:
                        btn.config(
                            text='',
                            image=self.yellow_chip_image,
                            bg=self.colors['empty'],  # Keep background neutral
                            state='disabled',
                            width=8,  # Maintain consistent width
                            height=4  # Maintain consistent height
                        )
                    else:
                        # Enhanced fallback for AI chips
                        btn.config(
                            text='●',  # Solid circle
                            image='',
                            font=('Arial', 18, 'bold'),  # Slightly smaller for consistency
                            bg='#f1c40f',   # Yellow background
                            fg='black',     # Black text for contrast
                            relief='raised',
                            bd=2,
                            activebackground='#f39c12',
                            state='disabled',
                            width=8,  # Maintain consistent width
                            height=4  # Maintain consistent height
                        )
        
        # Disable buttons for full columns
        if not self.game_over:
            legal_moves = self.board.get_legal_moves()
            for col in range(7):
                # Disable top button of full columns
                if col not in legal_moves:
                    for row in range(6):
                        if board_state[row, col] == 0:
                            self.buttons[row][col].config(state='disabled')
                            break
    
    def human_move(self, col: int) -> None:
        """Handle human player move."""
        if self.game_over or not self.board.is_legal_move(col):
            return
        
        # Make human move
        success = self.board.make_move(col, self.human_player)
        if not success:
            return
            
        self.update_display()
        
        # Check for game end
        if self.check_game_end():
            return
        
        # AI turn
        self.turn_label.config(text="AI is thinking...", fg='#f1c40f')
        self.root.update()
        
        # Delay for visual effect
        self.root.after(1000, self.ai_move)
    
    def ai_move(self) -> None:
        """Handle AI player move."""
        if self.game_over:
            return
            
        if self.ai_agent is None:
            # Human vs Human mode - wait for human input
            self.turn_label.config(
                text="Player 2's turn! Click a column to drop your piece.",
                fg='#f1c40f'
            )
            return
        
        # Get AI move
        legal_moves = self.board.get_legal_moves()
        if not legal_moves:
            return
            
        board_state = self.board.get_state()
        ai_action = self.ai_agent.choose_action(board_state, legal_moves)
        
        # Make AI move
        success = self.board.make_move(ai_action, self.ai_player)
        if success:
            self.update_display()
            
        # Check for game end
        if self.check_game_end():
            return
            
        # Back to human turn
        self.turn_label.config(
            text="Your turn! Click a column to drop your piece.",
            fg='#e74c3c'
        )
    
    def check_game_end(self) -> bool:
        """Check if game has ended and handle accordingly."""
        winner = self.board.check_winner()
        
        if winner is not None:
            self.game_over = True
            if winner == self.human_player:
                self.turn_label.config(
                    text="🎉 You won! Congratulations!",
                    fg='#27ae60'
                )
                messagebox.showinfo("Game Over", "Congratulations! You won!")
            else:
                self.turn_label.config(
                    text="🤖 AI won! Better luck next time!",
                    fg='#e74c3c'
                )
                messagebox.showinfo("Game Over", "AI won! Try again!")
            return True
            
        elif self.board.is_draw():
            self.game_over = True
            self.turn_label.config(
                text="🤝 It's a draw! Good game!",
                fg='#95a5a6'
            )
            messagebox.showinfo("Game Over", "It's a draw!")
            return True
            
        return False
    
    def new_game(self) -> None:
        """Start a new game with random starting player."""
        self.board.reset()
        self.game_over = False
        
        # Randomly assign who goes first for new game
        import random
        if random.random() < 0.5:
            self.human_player = 1  # Human goes first
            self.ai_player = 2     # AI goes second
            self.human_goes_first = True
            self.turn_label.config(
                text="Your turn! Click a column to drop your piece.",
                fg='#e74c3c'
            )
        else:
            self.human_player = 2  # Human goes second
            self.ai_player = 1     # AI goes first
            self.human_goes_first = False
            self.turn_label.config(
                text="AI goes first! Get ready...",
                fg='#f1c40f'
            )
            
        self.update_display()
        
        # Update AI agent's player_id if we have one
        if self.ai_agent:
            self.ai_agent.player_id = self.ai_player
            
        # If AI goes first, trigger AI move
        if not self.human_goes_first and self.ai_agent:
            self.root.after(1000, self.ai_move)
    
    def load_model_dialog(self) -> None:
        """Open dialog to load AI model."""
        from tkinter import filedialog
        
        # Check both new and old model directories
        initial_dirs = ["models_fixed", "models", "."]
        initial_dir = "."
        
        # Use the first directory that exists and has .pt files
        for dir_path in initial_dirs:
            if os.path.exists(dir_path):
                pt_files = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
                if pt_files:
                    initial_dir = dir_path
                    break
        
        model_path = filedialog.askopenfilename(
            title="Select AI Model",
            filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")],
            initialdir=initial_dir
        )
        
        if model_path:
            self.load_ai_agent(model_path)
            self.new_game()  # Start fresh with new AI
    
    def run(self) -> None:
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    """Main function to run the interactive game."""
    # Look for trained models
    model_path = None
    possible_paths = [
        "../models/double_dqn_final.pt",
        "../models/double_dqn_ep_52000.pt"
    ]
    
    for path in possible_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            model_path = full_path
            break
    
    if model_path is None:
        print("No trained model found. You can load one using the 'Load AI Model' button.")
    else:
        print(f"Loading AI model: {os.path.basename(model_path)}")
    
    # Create and run the GUI
    gui = Connect4GUI(model_path)
    gui.run()


if __name__ == "__main__":
    main()