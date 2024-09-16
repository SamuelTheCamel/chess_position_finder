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
        
        dist = 0.0
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
                dist += 50.0 # too few pieces (uh oh)
            else: # len(orig_squares) == len(target_squares)
                # calculate similarity between current piece arrangement and target piece arrangement
                for o_square in orig_squares:
                    min_dist = float("inf")
                    min_square = target_squares[0]

                    for t_square in target_squares:
                        curr_dist:float
                        if piece_type.lower() == "k":
                            curr_dist = chess.square_distance(o_square, t_square) * 2.0
                        elif piece_type.lower() == "q":
                            curr_dist = chess.square_distance(o_square, t_square) * 0.5
                        elif piece_type.lower() == "r":
                            curr_dist = chess.square_manhattan_distance(o_square, t_square) * 0.7
                        elif piece_type.lower() == "b":
                            o_card = (chess.square_file(o_square) + chess.square_rank(o_square)) % 2
                            t_card = (chess.square_file(t_square) + chess.square_rank(t_square)) % 2
                            if o_card == t_card:
                                curr_dist = chess.square_distance(o_square, t_square) * 1.0
                            else:
                                curr_dist = float("inf")
                        elif piece_type.lower() == "n":
                            curr_dist = chess.square_knight_distance(o_square, t_square) * 1.5
                        elif piece_type.lower() == "p":
                            file_dist = chess.square_file(t_square) - chess.square_file(o_square)
                            rank_dist = chess.square_rank(t_square) - chess.square_rank(o_square)
                            if rank_dist < 0:
                                curr_dist = float("inf")
                            else:
                                curr_dist = rank_dist * 1.2 + abs(file_dist) * 10.0
                        else:
                            curr_dist = float("inf")
                        
                        if curr_dist < min_dist:
                            min_dist = curr_dist
                            min_square = t_square
                    
                    dist += min_dist
                    target_squares.remove(min_square) # prevent multiple pieces from going to same square
        
        if dist < 1 and self.board.turn != self.target.turn:
            dist = 1.0

        self.dist = dist
        return dist
    
    def gen_children(self):
        '''
        Generates all child nodes of this node (positions after next move)
        Also updates self.dist
        '''
        move_list = self.board.legal_moves
        for move in move_list:
            # create new board with given move
            new_board = self.board.copy()
            new_board.push(move)
            # create new node with new board
            self.children.add(Eval_Node(new_board, self.target, self.depth + 1))
        
        self._update_dist()
    
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

    def __repr__(self):
        return f"Eval_Node: dist eval: {self.dist}; depth: {self.depth}; board: {self.board}"
    
    def __str__(self):
        return f"Eval_Node:\ndist eval: {self.dist}\ndepth: {self.depth}\n" + str(self.board)


def find(target:chess.Board, start:chess.Board = chess.Board(), max_depth=25, max_iter=1000, print_status=False) -> tuple[bool,list[chess.Move]]:
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
    leaves:list[Eval_Node] = [start_node]
    iter = 0

    while len(leaves) > 0:
        # sort leaves based on their eval distances
        leaves.sort(key = lambda x : x.dist_eval())
        # evaluate closest leaf to target
        current_node = leaves[0]
        # check if target is found or max_depth is reached
        if current_node.board == target:
            print("TARGET FOUND")
            return True, current_node.board.move_stack
        if current_node.depth == max_depth:
            print(f"MAX DEPTH REACHED\ndist: {current_node.dist_eval()}")
            return False, current_node.board.move_stack
        if iter == max_iter:
            print(f"MAX ITERATIONS REACHED\ndist: {current_node.dist_eval()}")
            return False, current_node.board.move_stack
        # generate child nodes
        current_node.gen_children()
        # add child nodes to leaves
        for node in current_node.children:
            leaves.append(node)
        # remove current_node from leaves
        del leaves[0]
        # print status
        if print_status:
            print(f"leaves: {len(leaves)}\ncurrent node:\n" + str(current_node))
        
        iter += 1

    print("NO LEAVES REMAINING")
    return False, []