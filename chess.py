import pygame
import copy
import sys
import random
from typing import List, Tuple, Optional, Dict

# ----- Constants and Colors -----
CELL_SIZE = 80
BOARD_SIZE = 8
WIDTH = CELL_SIZE * BOARD_SIZE + 200  # Extra space for sidebar
HEIGHT = CELL_SIZE * BOARD_SIZE
FPS = 60

# Colors
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT_COLOR = (247, 247, 105, 150)     # Semi-transparent yellow
SELECTED_COLOR = (247, 247, 105)           # Solid yellow for selected piece
MOVE_HINT_COLOR = (106, 168, 79, 150)      # Semi-transparent green
DANGER_HINT_COLOR = (220, 20, 60, 150)     # Semi-transparent red for threatened pieces
SIDEBAR_COLOR = (50, 50, 60)
TEXT_COLOR = (255, 255, 255)
HISTORY_COLOR = (70, 70, 80)

# Player colors with better contrast
PLAYER_COLORS = {
    0: (255, 70, 70),     # Player 1: Red (human)
    1: (70, 70, 255),     # Player 2: Blue (AI)
    2: (70, 200, 70),     # Player 3: Green (AI)
}

# Player constants
PLAYER1 = 0
PLAYER2 = 1
PLAYER3 = 2
PLAYERS = [PLAYER1, PLAYER2, PLAYER3]

# Standard chess piece values (for evaluation)
PIECE_VALUES = {
    'K': 1000,  # King (game over if lost)
    'Q': 9,
    'R': 5,
    'B': 3,
    'N': 3,
}

# Game settings
AI_DELAY = 500  # ms delay for AI moves
ANIMATION_DURATION = 300  # ms for move animation

# ----- Piece Class -----
class Piece:
    def __init__(self, piece_type: str, owner: int):
        self.type = piece_type  # 'K', 'Q', 'R', 'B', or 'N'
        self.owner = owner
        self.has_moved = False  # Track if piece has moved (for castling, pawns, etc.)

    def __repr__(self):
        return f"{self.type}{self.owner+1}"

# ----- Piece Drawing Function -----
def draw_piece(surface: pygame.Surface, rect: pygame.Rect, piece: Piece):
    """Draw a chess piece using shapes instead of Unicode symbols."""
    if piece.type == 'K':  # King
        pygame.draw.circle(surface, PLAYER_COLORS[piece.owner], rect.center, int(CELL_SIZE*0.3))
        pygame.draw.circle(surface, (0, 0, 0), rect.center, int(CELL_SIZE*0.3), 2)
        # Crown
        points = [
            (rect.centerx - CELL_SIZE*0.2, rect.centery - CELL_SIZE*0.1),
            (rect.centerx, rect.centery - CELL_SIZE*0.3),
            (rect.centerx + CELL_SIZE*0.2, rect.centery - CELL_SIZE*0.1),
            (rect.centerx + CELL_SIZE*0.15, rect.centery),
            (rect.centerx + CELL_SIZE*0.05, rect.centery - CELL_SIZE*0.15),
            (rect.centerx - CELL_SIZE*0.05, rect.centery - CELL_SIZE*0.15),
            (rect.centerx - CELL_SIZE*0.15, rect.centery)
        ]
        pygame.draw.polygon(surface, (255, 215, 0), points)
        pygame.draw.polygon(surface, (0, 0, 0), points, 2)
        
    elif piece.type == 'Q':  # Queen
        pygame.draw.circle(surface, PLAYER_COLORS[piece.owner], rect.center, int(CELL_SIZE*0.3))
        pygame.draw.circle(surface, (0, 0, 0), rect.center, int(CELL_SIZE*0.3), 2)
        # Crown with points
        for i in range(5):
            x_offset = (i - 2) * CELL_SIZE*0.1
            pygame.draw.polygon(surface, (255, 215, 0), [
                (rect.centerx + x_offset, rect.centery - CELL_SIZE*0.3),
                (rect.centerx + x_offset + CELL_SIZE*0.05, rect.centery - CELL_SIZE*0.15),
                (rect.centerx + x_offset - CELL_SIZE*0.05, rect.centery - CELL_SIZE*0.15)
            ])
        
    elif piece.type == 'R':  # Rook
        pygame.draw.rect(surface, PLAYER_COLORS[piece.owner], 
                        pygame.Rect(rect.centerx - CELL_SIZE*0.25, rect.centery - CELL_SIZE*0.25,
                                   CELL_SIZE*0.5, CELL_SIZE*0.5))
        pygame.draw.rect(surface, (0, 0, 0), 
                        pygame.Rect(rect.centerx - CELL_SIZE*0.25, rect.centery - CELL_SIZE*0.25,
                                   CELL_SIZE*0.5, CELL_SIZE*0.5), 2)
        # Battlements
        for i in range(4):
            x = rect.centerx - CELL_SIZE*0.2 + i * CELL_SIZE*0.13
            pygame.draw.rect(surface, (255, 215, 0), 
                            pygame.Rect(x, rect.centery - CELL_SIZE*0.3,
                                       CELL_SIZE*0.08, CELL_SIZE*0.1))
        
    elif piece.type == 'B':  # Bishop
        pygame.draw.polygon(surface, PLAYER_COLORS[piece.owner], [
            (rect.centerx, rect.centery - CELL_SIZE*0.3),
            (rect.centerx + CELL_SIZE*0.2, rect.centery + CELL_SIZE*0.3),
            (rect.centerx - CELL_SIZE*0.2, rect.centery + CELL_SIZE*0.3)
        ])
        pygame.draw.polygon(surface, (0, 0, 0), [
            (rect.centerx, rect.centery - CELL_SIZE*0.3),
            (rect.centerx + CELL_SIZE*0.2, rect.centery + CELL_SIZE*0.3),
            (rect.centerx - CELL_SIZE*0.2, rect.centery + CELL_SIZE*0.3)
        ], 2)
        # Mitre
        pygame.draw.rect(surface, (255, 215, 0), 
                        pygame.Rect(rect.centerx - CELL_SIZE*0.1, rect.centery - CELL_SIZE*0.35,
                                   CELL_SIZE*0.2, CELL_SIZE*0.15))
        
    elif piece.type == 'N':  # Knight
        # Horse head shape
        points = [
            (rect.centerx - CELL_SIZE*0.2, rect.centery + CELL_SIZE*0.2),
            (rect.centerx, rect.centery - CELL_SIZE*0.3),
            (rect.centerx + CELL_SIZE*0.2, rect.centery),
            (rect.centerx + CELL_SIZE*0.1, rect.centery + CELL_SIZE*0.3),
            (rect.centerx - CELL_SIZE*0.1, rect.centery + CELL_SIZE*0.2)
        ]
        pygame.draw.polygon(surface, PLAYER_COLORS[piece.owner], points)
        pygame.draw.polygon(surface, (0, 0, 0), points, 2)
        # Eye
        pygame.draw.circle(surface, (255, 255, 255), 
                          (rect.centerx + CELL_SIZE*0.05, rect.centery - CELL_SIZE*0.1), 
                          int(CELL_SIZE*0.05))
        pygame.draw.circle(surface, (0, 0, 0), 
                          (rect.centerx + CELL_SIZE*0.05, rect.centery - CELL_SIZE*0.1), 
                          int(CELL_SIZE*0.02))

# ----- Board Class -----
class Board:
    def __init__(self):
        self.size = BOARD_SIZE
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.move_history = []
        self.setup_board()

    def setup_board(self):
        """Custom layout for three players."""
        # Player 1 setup (bottom row)
        p1_row = 7
        p1_layout = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        for col, piece_type in enumerate(p1_layout):
            self.grid[p1_row][col] = Piece(piece_type, PLAYER1)
            
        # Player 2 setup (top row)
        p2_row = 0
        p2_layout = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        for col, piece_type in enumerate(p2_layout):
            self.grid[p2_row][col] = Piece(piece_type, PLAYER2)
        
        # Player 3 setup (left column, rows 1 to 6)
        p3_col = 0
        p3_layout = ['R', 'N', 'B', 'Q', 'K', 'R']
        for row, piece_type in zip(range(1, 7), p3_layout):
            self.grid[row][p3_col] = Piece(piece_type, PLAYER3)

    def is_player_defeated(self, player: int) -> bool:
        """Check if a player has been defeated (no king on board)."""
        for row in self.grid:
            for piece in row:
                if piece and piece.type == 'K' and piece.owner == player:
                    return False
        return True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, 
             selected_pos: Tuple[int, int] = None, valid_moves: List = None,
             show_threats: bool = False, current_player: int = None):
        """Draw the board grid and pieces with optional highlights."""
        # Draw squares
        for i in range(self.size):
            for j in range(self.size):
                color = LIGHT_SQUARE if (i+j) % 2 == 0 else DARK_SQUARE
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, color, rect)
                
                # Highlight selected square
                if selected_pos and selected_pos == (i, j):
                    pygame.draw.rect(surface, SELECTED_COLOR, rect, 3)
                
                # Highlight valid moves
                if valid_moves:
                    for move in valid_moves:
                        if move[1] == (i, j):
                            highlight_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                            highlight_surface.fill(MOVE_HINT_COLOR)
                            surface.blit(highlight_surface, rect)
                
                # Show threatened pieces (for current player)
                if show_threats and current_player is not None:
                    piece = self.grid[i][j]
                    if piece and piece.owner == current_player and self.is_under_threat(i, j, current_player):
                        threat_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                        threat_surface.fill(DANGER_HINT_COLOR)
                        surface.blit(threat_surface, rect)
                
                # Draw piece if present
                piece = self.grid[i][j]
                if piece:
                    draw_piece(surface, rect, piece)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def is_under_threat(self, row: int, col: int, player: int) -> bool:
        """Check if the piece at (row, col) is threatened by any opponent."""
        for i in range(self.size):
            for j in range(self.size):
                piece = self.grid[i][j]
                if piece and piece.owner != player and not self.is_player_defeated(piece.owner):
                    moves = self.get_moves_for_piece(i, j)
                    for (_, (r, c)) in moves:
                        if r == row and c == col:
                            return True
        return False

    def get_moves_for_piece(self, row: int, col: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Return a list of moves for the piece at (row, col)."""
        moves = []
        piece = self.grid[row][col]
        if not piece:
            return moves
        
        # Define helper to add moves along a direction for sliding pieces.
        def slide_moves(drow: int, dcol: int):
            r, c = row + drow, col + dcol
            while self.in_bounds(r, c):
                target = self.grid[r][c]
                if target is None:
                    moves.append(((row, col), (r, c)))
                else:
                    if target.owner != piece.owner and not self.is_player_defeated(target.owner):
                        moves.append(((row, col), (r, c)))
                    break
                r += drow
                c += dcol

        if piece.type == 'K':  # King
            directions = [(-1,-1), (-1,0), (-1,1),
                          (0,-1),          (0,1),
                          (1,-1),  (1,0),  (1,1)]
            for drow, dcol in directions:
                r, c = row + drow, col + dcol
                if self.in_bounds(r, c):
                    target = self.grid[r][c]
                    if target is None or (target.owner != piece.owner and not self.is_player_defeated(target.owner)):
                        moves.append(((row, col), (r, c)))
        elif piece.type == 'Q':  # Queen
            for drow, dcol in [(-1,-1), (-1,0), (-1,1),
                               (0,-1),          (0,1),
                               (1,-1),  (1,0),  (1,1)]:
                slide_moves(drow, dcol)
        elif piece.type == 'R':  # Rook
            for drow, dcol in [(-1,0), (1,0), (0,-1), (0,1)]:
                slide_moves(drow, dcol)
        elif piece.type == 'B':  # Bishop
            for drow, dcol in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                slide_moves(drow, dcol)
        elif piece.type == 'N':  # Knight
            knight_moves = [(2,1), (2,-1), (-2,1), (-2,-1),
                            (1,2), (1,-2), (-1,2), (-1,-2)]
            for drow, dcol in knight_moves:
                r, c = row + drow, col + dcol
                if self.in_bounds(r, c):
                    target = self.grid[r][c]
                    if target is None or (target.owner != piece.owner and not self.is_player_defeated(target.owner)):
                        moves.append(((row, col), (r, c)))
        return moves

    def get_all_moves(self, player: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Return all legal moves for the given player if they're not defeated."""
        if self.is_player_defeated(player):
            return []
        
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                piece = self.grid[i][j]
                if piece and piece.owner == player:
                    moves.extend(self.get_moves_for_piece(i, j))
        return moves

    def make_move(self, move: Tuple[Tuple[int, int], Tuple[int, int]]) -> 'Board':
        """Returns a new board state after applying the move."""
        new_board = copy.deepcopy(self)
        (src_row, src_col), (dest_row, dest_col) = move
        piece = new_board.grid[src_row][src_col]
        
        # Record the move in history (before and after positions)
        captured_piece = new_board.grid[dest_row][dest_col]
        move_record = {
            'player': piece.owner,
            'from': (src_row, src_col),
            'to': (dest_row, dest_col),
            'piece': piece.type,
            'captured': captured_piece.type if captured_piece else None
        }
        new_board.move_history.append(move_record)
        
        # Execute the move
        piece.has_moved = True
        new_board.grid[dest_row][dest_col] = piece
        new_board.grid[src_row][src_col] = None
        
        return new_board

    def is_game_over(self) -> Tuple[bool, Optional[int]]:
        """Game over if only one king remains or all but one player are defeated."""
        active_players = []
        for player in PLAYERS:
            if not self.is_player_defeated(player):
                active_players.append(player)
        
        if len(active_players) <= 1:
            return True, active_players[0] if active_players else None
        return False, None

# ----- Evaluation Heuristics -----
def evaluate_board(board: Board, player: int) -> float:
    """Evaluate the board state for the given player."""
    if board.is_player_defeated(player):
        return -float('inf')
    
    score = 0.0
    
    # Material score
    material = 0
    opponent_material = 0
    for i in range(board.size):
        for j in range(board.size):
            piece = board.grid[i][j]
            if piece and not board.is_player_defeated(piece.owner):
                value = PIECE_VALUES.get(piece.type, 0)
                if piece.owner == player:
                    material += value
                    # Bonus for central control
                    if (3 <= i <= 4) and (3 <= j <= 4):
                        material += 0.5
                else:
                    opponent_material += value
    
    # Mobility score (number of legal moves)
    mobility = len(board.get_all_moves(player)) * 0.1
    
    # King safety (penalty for being in center early game)
    king_pos = None
    for i in range(board.size):
        for j in range(board.size):
            piece = board.grid[i][j]
            if piece and piece.type == 'K' and piece.owner == player:
                king_pos = (i, j)
                break
    king_safety = 0
    if king_pos:
        # Penalize king being in center
        center_distance = max(abs(king_pos[0] - 3.5), abs(king_pos[1] - 3.5))
        king_safety = center_distance * 0.2
    
    # Combine factors with weights
    score = material - opponent_material * 0.5 + mobility + king_safety
    return score

# ----- AI: Multi-Player maxⁿ Algorithm -----
def maxn(board: Board, depth: int, current_player: int, num_players: int, 
         alpha: List[float] = None, beta: List[float] = None) -> Tuple[List[float], Optional[Tuple]]:
    """Multi-player minimax algorithm with alpha-beta pruning."""
    game_over, _ = board.is_game_over()
    if depth == 0 or game_over:
        return [evaluate_board(board, p) for p in range(num_players)], None

    if alpha is None:
        alpha = [-float('inf')] * num_players
    if beta is None:
        beta = [float('inf')] * num_players

    best_scores = None
    best_move = None
    moves = board.get_all_moves(current_player)
    
    if not moves:
        return [evaluate_board(board, p) for p in range(num_players)], None

    for move in moves:
        new_board = board.make_move(move)
        # Find next non-defeated player
        next_player = (current_player + 1) % num_players
        while new_board.is_player_defeated(next_player) and not game_over:
            next_player = (next_player + 1) % num_players
        
        scores, _ = maxn(new_board, depth - 1, next_player, num_players, alpha.copy(), beta.copy())
        
        if best_scores is None or scores[current_player] > best_scores[current_player]:
            best_scores = scores
            best_move = move
            alpha[current_player] = max(alpha[current_player], scores[current_player])
            
            # Alpha-beta pruning
            if any(alpha[p] > beta[p] for p in range(num_players)):
                break

    return best_scores, best_move

def ai_move(board: Board, player: int, depth: int = 2) -> Optional[Tuple]:
    """Get the AI's move with a given search depth."""
    if board.is_player_defeated(player):
        return None
    _, move = maxn(board, depth, player, len(PLAYERS))
    return move

# ----- GUI Components -----
def draw_sidebar(surface: pygame.Surface, font: pygame.font.Font, 
                 board: Board, current_player: int, game_over: bool, winner: Optional[int]):
    """Draw the sidebar with game information."""
    sidebar_rect = pygame.Rect(BOARD_SIZE * CELL_SIZE, 0, 200, HEIGHT)
    pygame.draw.rect(surface, SIDEBAR_COLOR, sidebar_rect)
    
    # Current player indicator
    turn_text = f"Player {current_player+1}'s turn" if not board.is_player_defeated(current_player) else "Game Over!"
    if game_over:
        turn_text = "Game Over!"
    turn_surf = font.render(turn_text, True, PLAYER_COLORS.get(current_player, TEXT_COLOR))
    surface.blit(turn_surf, (BOARD_SIZE * CELL_SIZE + 20, 20))
    
    # Winner announcement
    if game_over:
        winner_text = f"Winner: Player {winner+1}" if winner is not None else "Draw!"
        winner_surf = font.render(winner_text, True, TEXT_COLOR)
        surface.blit(winner_surf, (BOARD_SIZE * CELL_SIZE + 20, 60))
    
    # Player status (alive/dead)
    status_y = 100
    for player in PLAYERS:
        defeated = board.is_player_defeated(player)
        status = "DEFEATED" if defeated else "ACTIVE"
        color = (100, 100, 100) if defeated else PLAYER_COLORS.get(player, TEXT_COLOR)
        player_text = f"Player {player+1}: {status}"
        player_surf = font.render(player_text, True, color)
        surface.blit(player_surf, (BOARD_SIZE * CELL_SIZE + 20, status_y))
        status_y += 30
    
    # Move history
    history_rect = pygame.Rect(BOARD_SIZE * CELL_SIZE + 10, HEIGHT - 210, 180, 200)
    pygame.draw.rect(surface, HISTORY_COLOR, history_rect)
    history_title = font.render("Move History:", True, TEXT_COLOR)
    surface.blit(history_title, (BOARD_SIZE * CELL_SIZE + 20, HEIGHT - 200))
    
    # Display last 5 moves
    history_font = pygame.font.SysFont(None, 24)
    for i, move in enumerate(board.move_history[-5:]):
        move_text = f"P{move['player']+1}: {move['piece']} {chr(move['from'][1]+97)}{8-move['from'][0]}-{chr(move['to'][1]+97)}{8-move['to'][0]}"
        if move['captured']:
            move_text += f" (x{move['captured']})"
        move_surf = history_font.render(move_text, True, TEXT_COLOR)
        surface.blit(move_surf, (BOARD_SIZE * CELL_SIZE + 20, HEIGHT - 170 + i * 30))

def draw_promotion_menu(surface: pygame.Surface, font: pygame.font.Font, 
                        pos: Tuple[int, int], player: int) -> Dict[str, pygame.Rect]:
    """Draw promotion menu and return rects for each option."""
    x, y = pos
    menu_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE * 4)
    pygame.draw.rect(surface, SIDEBAR_COLOR, menu_rect)
    
    options = ['Q', 'R', 'B', 'N']
    option_rects = {}
    for i, piece_type in enumerate(options):
        rect = pygame.Rect(x, y + i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, (80, 80, 90), rect)
        piece = Piece(piece_type, player)
        draw_piece(surface, rect, piece)
        option_rects[piece_type] = rect
    
    return option_rects

# ----- Animation and Sound Effects -----
def animate_move(surface: pygame.Surface, board: Board, font: pygame.font.Font, 
                 move: Tuple[Tuple[int, int], Tuple[int, int]], 
                 moving_piece: Piece, duration: int = ANIMATION_DURATION):
    """Animate a piece moving from source to destination."""
    clock = pygame.time.Clock()
    start_ticks = pygame.time.get_ticks()
    (src_row, src_col), (dest_row, dest_col) = move
    start_x, start_y = src_col * CELL_SIZE, src_row * CELL_SIZE
    end_x, end_y = dest_col * CELL_SIZE, dest_row * CELL_SIZE

    while True:
        elapsed = pygame.time.get_ticks() - start_ticks
        if elapsed >= duration:
            break
        t = elapsed / duration
        current_x = start_x + (end_x - start_x) * t
        current_y = start_y + (end_y - start_y) * t

        # Redraw board
        board.draw(surface, font)
        
        # Draw the moving piece at the interpolated position
        temp_rect = pygame.Rect(current_x, current_y, CELL_SIZE, CELL_SIZE)
        draw_piece(surface, temp_rect, moving_piece)
        
        pygame.display.flip()
        clock.tick(FPS)

def play_sound(sound: Optional[pygame.mixer.Sound], sound_type: str = "move"):
    """Play a sound effect with optional variation."""
    if sound:
        if sound_type == "capture":
            sound.set_volume(0.7)
        else:
            sound.set_volume(0.5)
        sound.play()

# ----- Main Game Loop -----
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Three-Player Chess")
    clock = pygame.time.Clock()
    
    # Load fonts
    try:
        piece_font = pygame.font.Font("seguisym.ttf", 48)  # Try to load a font with chess symbols
    except:
        piece_font = pygame.font.SysFont("Arial", 48)  # Fallback
    ui_font = pygame.font.SysFont("Arial", 24)
    
    # Load sounds
    try:
        move_sound = pygame.mixer.Sound("move.wav")
        capture_sound = pygame.mixer.Sound("capture.wav")
    except:
        move_sound = None
        capture_sound = None

    # Game state
    board = Board()
    current_player = PLAYER1
    selected_piece = None
    valid_moves = []
    game_over = False
    winner = None
    show_threats = False
    promotion_move = None  # Stores move awaiting promotion choice
    
    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            # Handle key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:  # Toggle move hints
                    show_threats = not show_threats
                elif event.key == pygame.K_r:  # Reset game
                    board = Board()
                    current_player = PLAYER1
                    selected_piece = None
                    valid_moves = []
                    game_over = False
                    winner = None
                    promotion_move = None
            
            # Human player interactions
            if not game_over and current_player == PLAYER1 and event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Check if click is on promotion menu
                if promotion_move:
                    col, row = promotion_move[1]  # Destination of pawn move
                    if mouse_pos[0] > BOARD_SIZE * CELL_SIZE:  # Click in sidebar
                        promotion_rect = pygame.Rect(
                            BOARD_SIZE * CELL_SIZE + 20, 
                            row * CELL_SIZE, 
                            160, 
                            CELL_SIZE * 4
                        )
                        if promotion_rect.collidepoint(mouse_pos):
                            # Calculate which piece was selected
                            rel_y = mouse_pos[1] - row * CELL_SIZE
                            piece_index = rel_y // CELL_SIZE
                            promotion_pieces = ['Q', 'R', 'B', 'N']
                            if 0 <= piece_index < 4:
                                # Create the new piece
                                (src_row, src_col), (dest_row, dest_col) = promotion_move
                                new_piece = Piece(promotion_pieces[piece_index], current_player)
                                # Make the move
                                board.grid[src_row][src_col] = None
                                board.grid[dest_row][dest_col] = new_piece
                                # Find next non-defeated player
                                next_player = (current_player + 1) % len(PLAYERS)
                                while board.is_player_defeated(next_player) and not game_over:
                                    next_player = (next_player + 1) % len(PLAYERS)
                                current_player = next_player
                                promotion_move = None
                    continue
                
                # Normal move selection
                if mouse_pos[0] < BOARD_SIZE * CELL_SIZE:  # Click on board
                    row = mouse_pos[1] // CELL_SIZE
                    col = mouse_pos[0] // CELL_SIZE
                    
                    if not selected_piece:
                        # Select a piece if it belongs to the current player and player isn't defeated
                        piece = board.grid[row][col]
                        if piece and piece.owner == current_player and not board.is_player_defeated(current_player):
                            selected_piece = (row, col)
                            # Filter valid moves starting from this piece
                            all_moves = board.get_all_moves(current_player)
                            valid_moves = [m for m in all_moves if m[0] == selected_piece]
                    else:
                        # If a piece is already selected, check if the clicked cell is a valid destination
                        destination = (row, col)
                        move_chosen = None
                        for move in valid_moves:
                            if move[1] == destination:
                                move_chosen = move
                                break
                        
                        if move_chosen:
                            moving_piece = board.grid[move_chosen[0][0]][move_chosen[0][1]]
                            
                            # Check for pawn promotion (though we don't have pawns in this variant)
                            # This is kept as an example for future expansion
                            promote = False
                            if moving_piece.type == 'P':  # If we had pawns
                                if (moving_piece.owner == PLAYER1 and move_chosen[1][0] == 0) or \
                                   (moving_piece.owner == PLAYER2 and move_chosen[1][0] == 7) or \
                                   (moving_piece.owner == PLAYER3 and move_chosen[1][1] == 7):
                                    promote = True
                            
                            if promote:
                                promotion_move = move_chosen
                            else:
                                # Animate and make the move
                                animate_move(screen, board, piece_font, move_chosen, moving_piece)
                                
                                # Play appropriate sound
                                target_piece = board.grid[move_chosen[1][0]][move_chosen[1][1]]
                                if target_piece:
                                    play_sound(capture_sound, "capture")
                                else:
                                    play_sound(move_sound)
                                
                                board = board.make_move(move_chosen)
                                # Find next non-defeated player
                                next_player = (current_player + 1) % len(PLAYERS)
                                while board.is_player_defeated(next_player) and not game_over:
                                    next_player = (next_player + 1) % len(PLAYERS)
                                current_player = next_player
                        
                        selected_piece = None
                        valid_moves = []
        
        # AI moves
        if not game_over and current_player != PLAYER1 and promotion_move is None:
            if not board.is_player_defeated(current_player):
                pygame.time.delay(AI_DELAY)  # Brief delay for clarity
                move = ai_move(board, current_player, depth=2)
                if move:
                    moving_piece = board.grid[move[0][0]][move[0][1]]
                    animate_move(screen, board, piece_font, move, moving_piece)
                    
                    # Play appropriate sound
                    target_piece = board.grid[move[1][0]][move[1][1]]
                    if target_piece:
                        play_sound(capture_sound, "capture")
                    else:
                        play_sound(move_sound)
                    
                    board = board.make_move(move)
            
            # Find next non-defeated player
            next_player = (current_player + 1) % len(PLAYERS)
            while board.is_player_defeated(next_player) and not game_over:
                next_player = (next_player + 1) % len(PLAYERS)
            current_player = next_player
        
        # Check for game over condition
        if not game_over:
            game_over, winner = board.is_game_over()
        
        # ----- Drawing -----
        screen.fill((0, 0, 0))
        
        # Draw board with highlights
        board.draw(screen, piece_font, selected_piece, valid_moves, show_threats, current_player)
        
        # Draw sidebar
        draw_sidebar(screen, ui_font, board, current_player, game_over, winner)
        
        # Draw promotion menu if needed
        if promotion_move:
            _, (row, col) = promotion_move
            draw_promotion_menu(
                screen, piece_font, 
                (BOARD_SIZE * CELL_SIZE + 20, row * CELL_SIZE),
                current_player
            )
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
