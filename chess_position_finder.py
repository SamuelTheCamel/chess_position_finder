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
            print("yay")
            return 0.0

        dist = 0.0
        for file in range(8):
            for rank in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                target_piece = self.target.piece_at(square)
                if piece != target_piece:
                    dist += 4.0

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


def find(target:chess.Board, start:chess.Board = chess.Board(), max_depth=20, print_status=True) -> tuple[bool,list[chess.Move]]:
    '''
    Finds the target board from the start board.
    Set print_status to False to prevent find() from printing status messages.
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

    while len(leaves) > 0:
        # sort leaves based on their eval distances
        leaves.sort(key = lambda x : x.dist_eval())
        # evaluate closest leaf to target
        current_node = leaves[0]
        # check if target is found or max_depth is reached
        if current_node.board == target:
            return True, current_node.board.move_stack
        if current_node.depth == max_depth:
            return False, current_node.board.move_stack
        # generate child nodes
        current_node.gen_children()
        # this node is no longer a leaf, so delete it
        del leaves[0]

    return False, []