'''
Chess Position Finder
by Samuel Voltz

This program can find a set of moves from a given starting position to a given target position.
This program uses the chess module and the stockfish module (used to determine most likely player moves).
'''

import chess
import stockfish

class Eval_Node():

    def __init__(self, board:chess.Board, target:chess.Board):
        '''
        board: the board this node represents
        target: the target position we want to find
        '''
        self.board = board
        self.target = target
        self.children:set[Eval_Node] = set()
        self.dist:float|None = None # indicates that distance hasn't been evaluated yet

    def dist_eval(self) -> float:
        '''
        Returns the estimated distance from this node to the target
        '''

        if self.dist != None:
            return self.dist
        
        dist = 10.0
        # TODO
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
            self.children.add(Eval_Node(new_board, self.target))
        
        self._update_dist()
    
    def _update_dist(self):
        '''
        Helper function for updating distance evaluation based on child nodes
        '''
        
        min_dist:float = float("inf")
        for child in self.children:
            current_dist = child.dist_eval()
            if current_dist < min_dist:
                min_dist = current_dist
        
        self.dist = min_dist