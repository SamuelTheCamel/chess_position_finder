'''
Chess Position Finder
by Samuel Voltz

This program can find a set of moves from a given starting position to a given target position.
This program uses the chess module and the stockfish module (used to determine most likely player moves).
'''

import chess
import stockfish

# CHANGE THIS PATH TO LOCATION OF STOCKFISH EXE ON YOUR COMPUTER
sfish = stockfish.Stockfish("C:\\Users\\samue\\Documents\\stockfish\\stockfish-windows-x86-64-avx2.exe")
sfish.set_depth(2) # Low depth for faster computation

class Eval_Node():

    def __init__(self, board:chess.Board, target:chess.Board, depth:int = 0, max_depth:int=1000, prev_badness:float=0.0, prev_sf_eval:int=0, use_stockfish=True):
        '''
        board: the board this node represents
        target: the target position we want to find
        depth: the depth of this node in the tree
        max_depth: the depth at which this node will not generate any child nodes
        prev_badness: the badness of the parent node
        prev_sf_eval: the stockfish evaluation of the previous position
        use_stockfish: set to False to not use Stockfish for badness calculation (badness will default to 0)
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

    def dist_eval(self) -> float:
        '''
        Returns the estimated distance from this node to the target
        '''

        if self.dist != None:
            return self.dist
        
        if self.board == self.target:
            return 0.0

        if (self.board.outcome() != None and self.target.outcome() != None) or self.board.can_claim_threefold_repetition():
            return float("inf")
        
        for color in (chess.WHITE, chess.BLACK):
            if self.target.has_kingside_castling_rights(color) and not self.board.has_kingside_castling_rights(color):
                return float("inf")
            if self.target.has_queenside_castling_rights(color) and not self.board.has_queenside_castling_rights(color):
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
        
        # calcaulte similarity of certain piece types
        for piece_type in piece_freq:
            orig_squares = piece_freq[piece_type]
            target_squares = piece_freq_target[piece_type]

            if len(orig_squares) > len(target_squares):
                # too many pieces
                dist += 10.0 * (len(orig_squares) - len(target_squares))
            elif len(orig_squares) < len(target_squares):
                # too few pieces -> determine if more of that piece can possibly be created
                if piece_type in ("q", "r", "b", "n"):
                    if len(piece_freq["p"]) >= len(target_squares) - len(orig_squares):
                        dist += 50.0
                    else:
                        dist += float("inf")
                elif piece_type in ("Q", "R", "B", "N"):
                    if len(piece_freq["P"]) >= len(target_squares) - len(orig_squares):
                        dist += 50.0
                    else:
                        dist += float("inf")
                elif piece_type.lower() == "p":
                    dist += float("inf")
                else:
                    raise RuntimeError("invalid piece type has too many pieces in dist_eval()")
            else: # len(orig_squares) == len(target_squares)
                # calculate similarity between current piece arrangement and target piece arrangement
                for o_square in orig_squares:
                    min_dist = float("inf")
                    min_square = target_squares[0]

                    for t_square in target_squares:
                        curr_dist:float
                        if piece_type.lower() == "k":
                            curr_dist = chess.square_distance(o_square, t_square) * 1.5
                        elif piece_type.lower() == "q":
                            curr_dist = chess.square_distance(o_square, t_square) * 0.3
                        elif piece_type.lower() == "r":
                            curr_dist = chess.square_manhattan_distance(o_square, t_square) * 0.5
                        elif piece_type.lower() == "b":
                            o_card = (chess.square_file(o_square) + chess.square_rank(o_square)) % 2
                            t_card = (chess.square_file(t_square) + chess.square_rank(t_square)) % 2
                            if o_card == t_card:
                                curr_dist = chess.square_distance(o_square, t_square) * 0.5
                            else:
                                curr_dist = 100.0
                        elif piece_type.lower() == "n":
                            curr_dist = chess.square_knight_distance(o_square, t_square) * 1.5
                        elif piece_type.lower() == "p":
                            if piece_type == "P":
                                file_dist = chess.square_file(t_square) - chess.square_file(o_square)
                                rank_dist = chess.square_rank(t_square) - chess.square_rank(o_square)
                            else:
                                file_dist = chess.square_file(o_square) - chess.square_file(t_square)
                                rank_dist = chess.square_rank(o_square) - chess.square_rank(t_square)
                            # determine feasibility of pawn moves
                            if rank_dist < 0:
                                curr_dist = float("inf")
                            elif rank_dist == 0:
                                if file_dist == 0:
                                    curr_dist = 0.0
                                else:
                                    curr_dist = float("inf")
                            else: # rank_dist > 0
                                if abs(file_dist) > rank_dist:
                                    curr_dist = float("inf")
                                else:
                                    curr_dist = rank_dist * 1.0 + abs(file_dist)**2 * 10.0
                        else:
                            raise RuntimeError("unrecognized piece in dist_eval()")
                        
                        if curr_dist < min_dist:
                            min_dist = curr_dist
                            min_square = t_square
                    
                    dist += min_dist
                    target_squares.remove(min_square) # prevent multiple pieces from going to same square
        
        # near zero distance edge cases
        if dist < 1:
            # account for the position being the same but the turn being wrong
            if self.board.turn != self.target.turn:
                dist = 1.0
            # account for different castling rights
            for color in (chess.WHITE, chess.BLACK):
                if self.board.has_kingside_castling_rights(color) and not self.target.has_kingside_castling_rights(color):
                    dist = 1.0
                if self.board.has_queenside_castling_rights(color) and not self.target.has_queenside_castling_rights(color):
                    dist = 1.0

        if dist < 0:
            raise RuntimeError("negative distance value detected")

        self.dist = dist
        return dist
    
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
            # create new node with new board
            self.children.add(Eval_Node(new_board, self.target, self.depth + 1, self.max_depth,
                                        self.badness_eval(), self.sf_eval, self.use_stockfish))
        
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
    
    def precedence(self) -> float:
        '''
        Calculates the precedence of this node. Nodes with the lowest precedence will be evaluated first in minimax.
        (Can be negative)
        '''
        if self.use_stockfish:
            return self.dist_eval() + self.badness_eval()*0.25 - self.depth*0.25
        else:
            return self.dist_eval()

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


def find(target:chess.Board, start:chess.Board = chess.Board(), max_depth:int = 20, max_iter:int = 100, 
         print_status:bool = False, use_stockfish:bool = True) -> tuple[bool,list[chess.Move]]:
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
    '''
    if not(target.is_valid()):
        raise ValueError("invalid target board")
    if not(start.is_valid()):
        raise ValueError("invalid start board")
    
    start.clear_stack()
    target.clear_stack() # no cheating :)

    start_node = Eval_Node(start, target, max_depth=max_depth, use_stockfish=use_stockfish)
    leaves:list[Eval_Node] = [start_node] # sorted descending
    iter = 0

    while len(leaves) > 0:
        # evaluate closest leaf to target
        current_node = leaves.pop()
        # check if target is found or max_depth is reached
        if current_node.dist_eval() == 0:
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
            _insert_node_sorted(leaves, node)
        # print status
        if print_status:
            print(f"leaves: {len(leaves)}\ncurrent node:\n" + str(current_node))
        
        iter += 1

    if print_status:
        print("Final Node:\n" + str(current_node))
    print("NO LEAVES REMAINING")
    return False, []


def _insert_node_sorted(lst:list[Eval_Node], node:Eval_Node):
    '''
    Used to insert nodes into the sorted leaves list (descending order)
    Uses binary search
    '''
    node_prec = node.precedence()
    lower_bound:int = 0
    upper_bound:int = len(lst)
    guess:int

    while upper_bound > lower_bound + 1:
        guess = (lower_bound + upper_bound) // 2
        if lst[guess].precedence() < node_prec:
            upper_bound = guess
        elif lst[guess].precedence() > node_prec:
            lower_bound = guess
        else:
            lst.insert(guess, node)
            return
    
    lst.insert(upper_bound, node)