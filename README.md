# -An-AI-Driven-3-Player-Chess-Variant
# 🧠 Multi-Player Strategy Chess (Three-Player Chess Variant)
## 🎯 Project Overview

Multi-Player Strategy Chess is a creative reimagining of traditional chess, extended into a three-player variant with a unique board layout, custom rules, and intelligent AI opponents powered by the Maxⁿ algorithm.

# This Python-based game includes:
- A visually engaging graphical user interface built using **Pygame**
- Two AI players using refined evaluation heuristics
- Unique board mechanics supporting three-player competitive gameplay
- Smooth animations for an immersive experience

# 🧠 AI Strategy

The AI is implemented using the **Maxⁿ algorithm**, which extends the traditional minimax decision-making logic to a multi-agent environment. Key features include:

# - Evaluation Heuristics:
  - Piece value
  - Legal mobility
  - Central control bonuses

# - Game Tree Search:
  - Efficient turn-based decision-making across three players
  - Optional support for iterative deepening and alpha-beta pruning

## 🎮 Game Mechanics

# - Players:
  - Player 1: Human (bottom row)
  - Player 2: AI (top row)
  - Player 3: AI (left column)
- Pieces: King, Queen, Rook, Bishop, Knight  
- No Castling or En Passant  
- Elimination Rule: Player is eliminated if their king is captured  
- Winning Condition: Last king standing wins  
- Turn Order: Round-robin (P1 → P2 → P3 → repeat)

 # 🖼️ GUI Features
- Built with Pygame
- Board rendering with visual feedback
- Smooth piece animations
- Move validation highlights
- Sound effects on moves

 # 🧪 How to Run
git clone https:https://github.com/ShizaBajwa/-An-AI-Driven-3-Player-Chess-Variant.git
python chess.py
