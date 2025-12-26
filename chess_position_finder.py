'''
Chess Position Finder
by Samuel Voltz

This program can find a set of moves from a given starting position to a given target position.
This program uses the chess module and the stockfish module (used to determine most likely player moves).

IMPORTANT: Before running, make sure to:
1. Install Stockfish on your computer
2. Update stockfish_path.txt to the path where stockfish.exe is located

Create a board using the python chess module.
Use the function find(target, start) to find a target board from a starting board.
'''

import chess
import stockfish
import os
import sys
import heapq

# open stockfish_path.txt from the directory this python file is located in
file_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
file_loc = os.path.join(file_dir, "stockfish_path.txt")

if os.path.exists(file_loc):
    # read stockfish_path.txt
    with open(file_loc) as file:
        sfish_path = file.read()
else:
    # file does not exist, so prompt user for location
    # ask user for stockfish location
    sfish_path = input("Please enter the location of the Stockfish executable file (q to quit): ")
    
    if sfish_path == "q":
        sys.exit()

    # write to stockfish_path.txt
    with open(file_loc, "w") as file:
        file.write(sfish_path)

# setup stockfish
sfish = stockfish.Stockfish(sfish_path)
sfish.set_depth(2) # Low depth for faster computation

class Eval_Node():
    '''
    A node in the position finder search tree. Main contents include:
    self.board: the chess position this node represents
    self.target: the target chess position
    self.dist_eval(): the estimated distance (number of moves) from this node's board to the target
    self.badness_eval(): the badness of all the moves up to this point according to Stockfish
    self.children: the child nodes of this node (the next moves)
    self.gen_children(): generates the child nodes from this node

    NOTE: Make sure to clear the board_cache before each new search to ensure correct behavior
    '''

    board_cache:set[tuple[str,str]] = set()

    def __init__(self, board:chess.Board, target:chess.Board, depth:int = 0, max_depth:int=1000, 
                 prev_badness:float=0.0, prev_sf_eval:int=0, use_stockfish=True, 
                 skill:float=1.0, depth_reward:float=0.25):
        '''
        board: the board this node represents
        target: the target position we want to find
        depth: the depth of this node in the tree
        max_depth: the depth at which this node will not generate any child nodes (for better results, set this to be 2x the difference in turn number)
        prev_badness: the badness of the parent node
        prev_sf_eval: the stockfish evaluation of the previous position
        use_stockfish: set to False to not use Stockfish for badness calculation (badness will default to 0)
        skill: determines how much influence Stockfish will have
        depth_reward: determines how much the search algorithm will prioritize deeper nodes (set to negative value to make it prioritize closer nodes)
        '''
        self.board = board
        self.target = target
        self.children:set[Eval_Node] = set()
        self.dist:float|None = None # None when distance hasn't been evaluated yet
        self.depth = depth
        self.max_depth = max_depth

        self.prev_badness = prev_badness
        self.prev_sf_eval = prev_sf_eval
        self.badness:float|None = None # None when badness hasn't been evaluated yet
        self.sf_eval:int = prev_sf_eval # initialize to prev_sf_eval as failsafe
        self.use_stockfish = use_stockfish

        self.skill = skill
        self.depth_reward = depth_reward

        # add this node to board cache
        reduced_fen_board = reduce_fen(self.board.fen())
        reduced_fen_target = reduce_fen(self.board.fen())
        self.board_cache.add((reduced_fen_board, reduced_fen_target))


    def dist_eval(self) -> float:
        '''
        Returns the estimated distance from this node to the target
        '''

        if self.dist != None:
            return self.dist
        
        if board_equals(self.board, self.target):
            return 0.0

        if (self.board.outcome() != None and self.target.outcome() != None) or self.board.can_claim_threefold_repetition():
            return float("inf")
        
        for color in (chess.WHITE, chess.BLACK):

            # check castling rights
            if self.target.has_kingside_castling_rights(color) and not self.board.has_kingside_castling_rights(color):
                return float("inf")
            if self.target.has_queenside_castling_rights(color) and not self.board.has_queenside_castling_rights(color):
                return float("inf")
        
            # check piece counts
            if self.piece_count(self.board, color) < self.piece_count(self.target, color):
                return float("inf")

        dist:float = 0.0

        piece_freq:dict[str,list[chess.Square]] = {"P":[], "p":[], "N":[], "n":[], "B":[], "b":[], 
                                                   "R":[], "r":[], "Q":[], "q":[], "K":[], "k":[]}
        piece_freq_target:dict[str,list[chess.Square]] = {"P":[], "p":[], "N":[], "n":[], "B":[], "b":[], 
                                                          "R":[], "r":[], "Q":[], "q":[], "K":[], "k":[]}

        for file in range(8):
            for rank in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                target_piece = self.target.piece_at(square)

                if piece != None:
                    piece_freq[str(piece)].append(square)
                if target_piece != None:
                    piece_freq_target[str(target_piece)].append(square)

                if piece != target_piece:
                    dist += 0.1
        
        # sort pawn targets to prevent incorrect targeting issues
        piece_freq_target["p"].sort(reverse=True)
        piece_freq_target["P"].sort()
        
        # calculate how close pieces are to their target squares
        # by looking at each target square and adding the distance of the closest orig square
        # and then removing that square to prevent two pieces moving towards the same target
        for piece_type in piece_freq_target:
            target_squares:list[chess.Square] = piece_freq_target[piece_type]
            
            same_type_squares:list[chess.Square] = piece_freq[piece_type]
            prom_squares:list[chess.Square]
            if piece_type in ("n","b","r","q"):
                prom_squares =  piece_freq["p"]
            elif piece_type in ("N","B","R","Q"):
                prom_squares =  piece_freq["P"]
            else:
                prom_squares = []
            
            orig_squares:list[chess.Square] = same_type_squares + prom_squares

            for t_square in target_squares:

                if len(orig_squares) == 0:
                    return float("inf") # no remaining pieces that can reach this target
                
                min_dist:float = float("inf")
                min_square:chess.Square = orig_squares[0]
                
                for o_square in orig_squares:
                    target_type_obj = self.target.piece_at(t_square)
                    orig_type_obj = self.board.piece_at(o_square)
                    
                    if target_type_obj == None or orig_type_obj == None: # this is to make type checking happy
                        raise RuntimeError("o_square or t_square has no piece in dist_eval()")
                    
                    curr_dist = self.piece_dist(orig_type_obj, o_square, target_type_obj, t_square)

                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        min_square = o_square

                dist += min_dist

                # the orig piece for this target piece has been found, so remove it
                orig_squares.remove(min_square) 
                if min_square in same_type_squares:
                    same_type_squares.remove(min_square)
                if min_square in prom_squares:
                    prom_squares.remove(min_square)
        
        # check for orig pieces that are left over (pieces that need to be captured)
        for piece_type in piece_freq:
            orig_squares = piece_freq[piece_type]

            attack_color:chess.Color
            if piece_type == piece_type.lower():
                attack_color = chess.WHITE # piece is black
            else:
                attack_color = chess.BLACK # piece is white

            for o_square in orig_squares:
                if self.board.is_attacked_by(attack_color, o_square):
                    dist += 3.0 # lower distance if being attacked
                else:
                    dist += 5.0

        if dist < 0:
            raise RuntimeError("negative distance value detected")

        # account for slight mismatches (board equivalence was already checked)
        if dist < 1:
            dist = 1.0

        self.dist = dist
        return dist
    

    def piece_dist(self, current_type:chess.Piece, current_square:chess.Square, 
                   target_type:chess.Piece, target_square:chess.Square) -> float:
        '''
        Returns the estimated distance between a piece's current square and a target square.
        Please also speicify the current piece's type and the target piece type (may be different in the case of promotion).
        '''
        curr_str = str(current_type)
        target_str = str(target_type)

        if curr_str.lower() == "p":

            # pawn -> pawn
            if curr_str == target_str:
                # get file and rank distances (reversed based on color)
                if curr_str == "P":
                    file_dist = chess.square_file(target_square) - chess.square_file(current_square)
                    rank_dist = chess.square_rank(target_square) - chess.square_rank(current_square)
                else:
                    file_dist = chess.square_file(current_square) - chess.square_file(target_square)
                    rank_dist = chess.square_rank(current_square) - chess.square_rank(target_square)
                
                if rank_dist < 0: # pawn too far
                    return float("inf") 
                elif rank_dist == 0: # pawn at correct rank
                    if file_dist == 0:
                        return 0.0 # same position
                    else:
                        return float("inf") # wrong file
                else: # rank_dist > 0

                    if abs(file_dist) > rank_dist:
                        return float("inf") # impossible to reach with diagonal moves
                    
                    # check number of captures available
                    current_num_pieces = self.piece_count(self.board, not(current_type.color))
                    target_num_pieces = self.piece_count(self.target, not(current_type.color))
                    num_captures = current_num_pieces - target_num_pieces
                    if abs(file_dist) > num_captures:
                        return float("inf") # not enough captures available

                    return rank_dist * 1.0 + abs(file_dist)**2 * 10.0
            
            # pawn -> other piece (white)
            if curr_str == "P" and target_str in ("N","B","R","Q"):
                prom_dist = 7 - chess.square_rank(current_square)
                return 15.0 + prom_dist * 10.0
            
            # pawn -> other piece (black)
            if curr_str == "p" and target_str in ("n","b","r","q"):
                prom_dist = chess.square_rank(current_square)
                return 15.0 + prom_dist * 10.0
            
            # pawn -> king (or error case)
            return float("inf")
        
        # check if there is piece mismatch
        if curr_str != target_str:
            return float("inf")

        if curr_str.lower() == "n":
            return chess.square_knight_distance(current_square, target_square) * 1.5
        
        if curr_str.lower() == "b":
            # check square colors
            c_card = (chess.square_file(current_square) + chess.square_rank(current_square)) % 2
            t_card = (chess.square_file(target_square) + chess.square_rank(target_square)) % 2
            if c_card == t_card:
                return chess.square_distance(current_square, target_square) * 0.5
            else:
                return float("inf") # opposite colored squares
        
        if curr_str.lower() == "r":
            return chess.square_manhattan_distance(current_square, target_square) * 0.5
        
        if curr_str.lower() == "q":
            return chess.square_distance(current_square, target_square) * 0.3
        
        if curr_str.lower() == "k":
            return chess.square_distance(current_square, target_square) * 1.5

        raise ValueError("invalid piece type in piece_dist()")
    

    def piece_count(self, board:chess.Board, color:chess.Color) -> int:
        '''
        Counts the total number of pieces of a certain color in a given board.
        Includes the king.
        '''
        count = 0
        for file in range(8):
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece != None and piece.color == color:
                    count += 1
        return count

    
    def gen_children(self):
        '''
        Generates all child nodes of this node (positions after next move)
        Does nothing if depth = max_depth
        '''
        if self.depth == self.max_depth:
            return
        move_list = self.board.legal_moves
        for move in move_list:
            # create new board with given move
            new_board = self.board.copy()
            new_board.push(move)
            # check if board has already been visited
            reduced_fen_new_board = reduce_fen(new_board.fen())
            reduced_fen_target = reduce_fen(self.board.fen())
            if (reduced_fen_new_board, reduced_fen_target) in self.board_cache:
                continue
            # create new node with new board
            self.children.add(Eval_Node(new_board, self.target, self.depth + 1, self.max_depth,
                                        self.badness_eval(), self.sf_eval, self.use_stockfish, 
                                        self.skill, self.depth_reward))
        
        #self._update_dist()
    
    def _update_dist(self):
        '''
        Helper function for updating distance evaluation based on child nodes
        '''
        
        if self.dist == 0.0:
            return # position already found

        min_dist:float = float("inf")
        for child in self.children:
            current_dist = child.dist_eval()
            if current_dist < min_dist:
                min_dist = current_dist
        
        self.dist = min_dist + 1.0
    
    def priority(self) -> float:
        '''
        Calculates the priority of this node. Lower values indicate higher priority.
        Nodes with the lowest value will be evaluated first in minimax.
        (Can be negative)
        '''
        if self.use_stockfish:
            return self.dist_eval() + self.badness_eval()*self.skill - self.depth*self.depth_reward
        else:
            return self.dist_eval() + 0.25 - self.depth*self.depth_reward

    def badness_eval(self) -> float:
        '''
        Returns the badness of the set of moves leading to this position as determined by Stockfish
        (Used to determine most likely human play)
        '''

        if self.badness != None:
            return self.badness
        
        if self.use_stockfish:
            sfish.set_fen_position(self.board.fen())
            sf_eval_dict = sfish.get_evaluation()
            if sf_eval_dict["type"] == "cp":
                self.sf_eval = sf_eval_dict["value"]
            else:
                if sf_eval_dict["value"] > 0:
                    self.sf_eval = 1000
                elif sf_eval_dict["value"] < 0:
                    self.sf_eval = -1000
                else:
                    self.sf_eval = 0 # idk if this is possible or not
            
            self.badness = self.prev_badness + abs(self.sf_eval - self.prev_sf_eval)*0.01

        else:
            self.badness = 0.0 # stockfish disabled
            self.sf_eval = 0

        return self.badness

    def __repr__(self):
        return f"Eval_Node: dist eval: {self.dist}; badness: {self.badness}; depth: {self.depth}; board: {repr(self.board)}"
    
    def __str__(self):
        return f"Eval_Node:\ndist eval: {self.dist}\nbadness: {self.badness}\ndepth: {self.depth}\n" + str(self.board)

    # Less than and greater than dunders (used for binary heap)

    def __lt__(self, other: 'Eval_Node') -> bool:
        return self.priority() < other.priority()

    def __gt__(self, other: 'Eval_Node') -> bool:
        return self.priority() > other.priority()

    @staticmethod
    def clear_cache():
        '''
        Clears the board cache. This should be called at the start/end of any search.
        '''
        Eval_Node.board_cache = set()


def find(target:chess.Board, start:chess.Board = chess.Board(), max_depth:int = 30, max_iter:int = 100, 
         print_status:bool = False, use_stockfish:bool = True, 
         skill:float=1.0, depth_reward:float=0.25) -> tuple[bool,list[chess.Move]]:
    '''
    Finds the target board from the start board.
    Set print_status to True to see status messages.
    Returns True and the list of moves used to reach the target if successful.
    Returns False and an empty list if unsuccessful.
    Raises a ValueError if the start or target board are invalid.

    target: the board to find
    start: the board to start from
    max_depth: the maximum depth this function will search to
    max_iter: the maximum number of iterations this function can take
    print_status: set to True to see status messages during calculation
    use_stockfish: set to False to not use Stockfish during evaluation (makes it way faster)
    skill: determines how well both sides should play according to Stockfish
    depth_reward: determines how much the search algorithm will prioritize deeper nodes (set to negative value to make it prioritize closer nodes)
    '''
    if not(target.is_valid()):
        raise ValueError("invalid target board")
    if not(start.is_valid()):
        raise ValueError("invalid start board")
    
    start.clear_stack()
    target.clear_stack() # no cheating :)

    Eval_Node.clear_cache() # clear cached info from previous search

    start_node = Eval_Node(start, target, max_depth=max_depth, use_stockfish=use_stockfish, 
                           skill=skill, depth_reward=depth_reward)
    leaves:list[Eval_Node] = [start_node] # must maintain min-heap structure using heapq
    iter = 0

    while len(leaves) > 0:
        # evaluate closest leaf to target
        current_node = heapq.heappop(leaves)
        # check if target is found or max_depth is reached
        if board_equals(current_node.board, target):
            if print_status:
                print("Final Node:\n" + str(current_node))
                print(f"TARGET FOUND\niterations: {iter}")
            return True, current_node.board.move_stack
        if iter == max_iter:
            if print_status:
                print("Final Node:\n" + str(current_node))
                print(f"MAX ITERATIONS REACHED\ndist eval: {current_node.dist_eval()}")
            return False, []
        # generate child nodes
        current_node.gen_children()
        # add child nodes to leaves
        for node in current_node.children:
            heapq.heappush(leaves, node)
        # print status
        if print_status:
            print(f"leaves: {len(leaves)}\ncurrent node:\n" + str(current_node))
        
        iter += 1

    if print_status:
        print("Final Node:\n" + str(current_node))
        print("NO MOVES REMAINING")
    return False, []


def reduce_fen(fen:str) -> str:
    '''
    Removes the turn number and 50-move-rule information from the FEN string.
    '''
    fen_list = fen.split(" ")
    new = " ".join(fen_list[:4])
    return new


def board_equals(board1:chess.Board, board2:chess.Board):
    '''
    Returns True if board1 is the same as board2 except for turn number and 50-move-rule tracking.
    '''
    return reduce_fen(board1.fen()) == reduce_fen(board2.fen())