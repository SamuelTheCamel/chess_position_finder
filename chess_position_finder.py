'''
Chess Position Finder
by Samuel Voltz

This program can find a set of moves from a given starting position to a given target position.
This program uses the chess module and the stockfish module (used to determine most likely player moves).
'''

import chess
import stockfish

class Eval_Node():

    def __init__(self, board:chess.Board, target:chess.Board, depth:int = 0):
        '''
        board: the board this node represents
        target: the target position we want to find
        '''
        self.board = board
        self.target = target
        self.children:set[Eval_Node] = set()
        self.dist:float|None = None # indicates that distance hasn't been evaluated yet
        self.depth = depth

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
                dist += 5.0 # too many pieces
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
                                curr_dist = chess.square_distance(o_square, t_square) * 0.7
                            else:
                                curr_dist = 100.0
                        elif piece_type.lower() == "n":
                            curr_dist = chess.square_knight_distance(o_square, t_square) * 1.5
                        elif piece_type == "P":
                            file_dist = chess.square_file(t_square) - chess.square_file(o_square)
                            rank_dist = chess.square_rank(t_square) - chess.square_rank(o_square)
                            if rank_dist < 0:
                                curr_dist = float("inf")
                            else:
                                curr_dist = rank_dist * 1.2 + abs(file_dist) * 10.0
                        elif piece_type == "p":
                            file_dist = chess.square_file(o_square) - chess.square_file(t_square)
                            rank_dist = chess.square_rank(o_square) - chess.square_rank(t_square)
                            if rank_dist < 0:
                                curr_dist = float("inf")
                            else:
                                curr_dist = rank_dist * 1.2 + abs(file_dist) * 10.0
                        else:
                            raise RuntimeError("unrecognized piece in dist_eval()")
                        
                        if curr_dist < min_dist:
                            min_dist = curr_dist
                            min_square = t_square
                    
                    dist += min_dist
                    target_squares.remove(min_square) # prevent multiple pieces from going to same square
        
        if dist < 1 and self.board.turn != self.target.turn:
            dist = 1.0

        if dist < 0:
            raise RuntimeError("negative distance value detected")

        self.dist = dist
        return dist
    
    def gen_children(self):
        '''
        Generates all child nodes of this node (positions after next move)
        '''
        move_list = self.board.legal_moves
        for move in move_list:
            # create new board with given move
            new_board = self.board.copy()
            new_board.push(move)
            # create new node with new board
            self.children.add(Eval_Node(new_board, self.target, self.depth + 1))
        
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
    
    def precedence(self):
        '''
        Calculates the precedence of this node. Nodes with the lowest precedence will be evaluated first in minimax.
        '''
        return self.dist_eval() + self.depth * 0.1

    def __repr__(self):
        return f"Eval_Node: dist eval: {self.dist}; depth: {self.depth}; board: {repr(self.board)}"
    
    def __str__(self):
        return f"Eval_Node:\ndist eval: {self.dist}\ndepth: {self.depth}\n" + str(self.board)


def find(target:chess.Board, start:chess.Board = chess.Board(), max_depth=20, max_iter=500, print_status=False) -> tuple[bool,list[chess.Move]]:
    '''
    Finds the target board from the start board.
    Set print_status to True to see status messages.
    Returns True and the list of moves used to reach the target if successful.
    Returns False and the list of moves to the closest position if target is not reached in max_depth moves.
    Returns False and an empty list if all moves lead to stalemate/checkmate.
    Raises a ValueError if the start or target board are invalid.
    '''
    if not(target.is_valid()):
        raise ValueError("invalid target board")
    if not(start.is_valid()):
        raise ValueError("invalid start board")
    
    start.clear_stack()
    target.clear_stack() # no cheating :)

    start_node = Eval_Node(start, target)
    leaves:list[Eval_Node] = [start_node] # sorted descending
    iter = 0

    while len(leaves) > 0:
        # evaluate closest leaf to target
        current_node = leaves.pop()
        # check if target is found or max_depth is reached
        if current_node.dist_eval() == 0:
            print(f"TARGET FOUND\niterations: {iter}")
            return True, current_node.board.move_stack
        if current_node.depth == max_depth:
            print(f"MAX DEPTH REACHED\ndist eval: {current_node.dist_eval()}")
            return False, current_node.board.move_stack
        if iter == max_iter:
            print(f"MAX ITERATIONS REACHED\ndist eval: {current_node.dist_eval()}")
            return False, current_node.board.move_stack
        # generate child nodes
        current_node.gen_children()
        # add child nodes to leaves
        for node in current_node.children:
            _insert_node_sorted(leaves, node)
        # print status
        if print_status:
            print(f"leaves: {len(leaves)}\ncurrent node:\n" + str(current_node))
        
        iter += 1

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
            lst.insert(guess + 1, node)
            return
    
    lst.insert(upper_bound, node)