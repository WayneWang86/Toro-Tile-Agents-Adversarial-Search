'''
wyf9686_TTS_agent.py
modified from a version of starter code PlayerSkeleton.py, in CSE 415
by Wayne Wang
Date: Nov 08, 2019

Assignment 5, in CSE 415, Autumn 2019.

STUDENTS: IMPLEMENTATION OF wyf9686_TTS_agent.py BY MODIFYING THE STARTER CODE
PlayerSkeleton.py, FROM CSE 415.

'''

from TTS_State import TTS_State
import time

USE_CUSTOM_STATIC_EVAL_FUNCTION = True

K = 0
Board_Rows = 0
Board_Cols = 0
Directions = [(1, 0), (1, -1), (0, 1), (1, 1)]
if_block = 0
Alpha_beta = True
ply_covered = []
My_Side = ''
MAX_DEPTH_REACHED = 0
n_states_expanded = 0
n_static_evals = 0
n_cutoffs = 0
max_depth = 0
start_time = None
total_move = 0

utterances = ["You get some swag, let's see my move", "Is this a rational move?",
              "Smart move, but don't forget I am the best defender", "Look at me, I am reading your mind",
              "Well, well, well...", "You did well man", "Nice move, but I am better", "I am winning, be careful",
              "Man, you are good", "You think you are smarter than me?", "It's time to show you my true skills",
              "You are losing!"]



class MY_TTS_State(TTS_State):
    def static_eval(self):
        if USE_CUSTOM_STATIC_EVAL_FUNCTION:
            return self.custom_static_eval()
        else:
            return self.basic_static_eval()

    def basic_static_eval(self):
        # raise Exception("basic_static_eval not yet implemented.")
        curr_board = self.board
        twf = 0
        tbf = 0
        for row in range(Board_Rows):
            for col in range(Board_Cols):
                if curr_board[row][col] == 'W':
                    twf += check_score(curr_board, Board_Rows, Board_Cols, row, col)
                elif curr_board[row][col] == 'B':
                    tbf += check_score(curr_board, Board_Rows, Board_Cols, row, col)
        value = twf - tbf
        return value

    def custom_static_eval(self):
        # raise Exception("custom_static_eval not yet implemented.")
        global Directions, if_block
        if_block = 0
        board = self.board
        height = len(board)  # height of the board
        width = len(board[0])  # width of the board
        win_w = 0
        win_b = 0
        w_eval = 0
        b_eval = 0
        for i in range(height):
            for j in range(width):
                if board[i][j] != '-':
                    for direct in Directions:
                        i_c = i
                        j_c = j
                        num_w = 0
                        num_b = 0
                        num_f = 0
                        i_move = direct[0]
                        j_move = direct[1]
                        if board[i_c][j_c] == 'W':  # white counter
                            num_w += 1
                        if board[i_c][j_c] == 'B':  # black counter
                            num_b += 1
                        if board[i_c][j_c] == '-':  # forbidden counter
                            num_f += 1
                            continue
                        for k in range(K - 1):
                            i_c += i_move
                            j_c += j_move
                            if i_c < 0 or i_c >= height:  # L-R case
                                i_c = ((i_c + height) % height)
                            if j_c < 0 or j_c >= width:  # T-B case
                                j_c = ((j_c + width) % width)
                            if board[i_c][j_c] == 'W':  # white counter
                                num_w += 1
                            if board[i_c][j_c] == 'B':  # black counter
                                num_b += 1
                            if board[i_c][j_c] == '-':
                                num_f += 1
                        if (num_w > 0 and num_b > 0) or num_f > 0:  # block situation
                            continue
                        if num_w == int(K / 2):
                            w_eval += 1
                        if num_b == int(K / 2):
                            b_eval += 1
                        if num_w >= K - 1:
                            w_eval += 100
                            return 100
                        if num_b >= K - 1:
                            b_eval += 100
                            return -100
                win_w += w_eval
                win_b += b_eval
        win_sum = win_w - win_b
        return win_sum


def check_score(board, nrow, ncol, row, col):
    score = 0
    rows = list(range(nrow))
    cols = list(range(ncol))
    pre_row = rows[row - 1]
    pre_col = cols[col - 1]
    next_row = (row + 1) % nrow
    next_col = (col + 1) % ncol

    for r in [pre_row, row, next_row]:
        for c in [pre_col, col, next_col]:
            if board[r][c] == ' ':
                score += 1
    return score


# The following is a skeleton for the function called parameterized_minimax,
# which should be a top-level function in each agent file.
# A tester or an autograder may do something like
# import ABC_TTS_agent as player, call get_ready(),
# and then it will be able to call tryout using something like this:
# results = player.parameterized_minimax(**kwargs)

def parameterized_minimax(
        current_state = None,
        max_ply = 2,
        alpha_beta = False,
        use_custom_static_eval_function = False,
        iterative_deepening = False,
        time_limit = 1.0):
    # All students, add code to replace these default
    # values with correct values from your agent (either here or below).

    global Alpha_beta, n_states_expanded, n_static_evals, n_cutoffs, My_Side, \
            USE_CUSTOM_STATIC_EVAL_FUNCTION, start_time

    current_state.__class__ = MY_TTS_State
    n_states_expanded = 0
    n_static_evals = 0
    n_cutoffs = 0

    if use_custom_static_eval_function:
        USE_CUSTOM_STATIC_EVAL_FUNCTION = True

    if (current_state != None):
        current_state.__class__ = MY_TTS_State
        current_state_static_val = current_state.static_eval()

    start_time = time.time()

    Alpha_beta = alpha_beta

    # if alpha_beta == True:
    #     current_state_static_val = _alpha_beta(current_state, max_ply, 0, -100000, 100000, My_Side, time_limit)
    # else:
    #     # current_state_static_val = _minimax(current_state, My_Side, max_ply, 0, time_limit)
    #     current_state_static_val = _minimax(current_state, My_Side, max_ply,time_limit)

    if (iterative_deepening):
        IDDFS(current_state, current_state.whose_turn, max_ply, time_limit)

    DATA = {}
    DATA['CURRENT_STATE_STATIC_VAL'] = current_state_static_val
    DATA['N_STATES_EXPANDED'] = n_states_expanded
    DATA['N_STATIC_EVALS'] = n_static_evals
    DATA['N_CUTOFFS'] = n_cutoffs
    return (DATA)

def _minimax (curr_state, side, plyLeft, time_limit):
    global start_time
    curr_state.__class__ = MY_TTS_State
    curr_board = curr_state.board
    all_vacancy = _find_all_vacancy(curr_board)
    if all_vacancy == []:
        return [[], curr_state.static_eval()]
    location = all_vacancy[0]
    if plyLeft == 0:
        return [location, curr_state.static_eval()]
    otherMove = 'W'
    if side == 'W':
        provisional = -1000000
        otherMove = 'B'
    else:
        provisional = 1000000
    for vac in all_vacancy:
        newState = curr_state.copy()
        newBoard = newState.board
        newBoard[vac[0]][vac[1]] = side
        newVal = _minimax(newState, otherMove, plyLeft - 1, time_limit)[1]
        if (side == 'W' and newVal > provisional or side == 'B' and newVal < provisional):
            provisional = newVal
            location = vac
        if (time.perf_counter() - start_time > time_limit):
            break
    return [location, provisional]


def _alpha_beta(curr_state, side, alpha, beta, plyLeft, time_limit):
    global start_time, n_states_expanded, max_depth, n_cutoffs, ply_covered
    if plyLeft not in ply_covered:
        max_depth += 1
    curr_state.__class__ = MY_TTS_State
    curr_board = curr_state.board
    all_vacancy = _find_all_vacancy(curr_board)
    if (all_vacancy == []):
        return [[], curr_state.static_eval()]
    move = all_vacancy[0]
    if plyLeft == 0:
        return [move, curr_state.static_eval()]
    otherMove = 'W'
    if side == 'W':
        otherMove = 'B'
    # print("in AB")
    n_states_expanded += 1
    if (side == 'W'):
        for vac in all_vacancy:
            newState = curr_state.copy()
            newBoard = newState.board
            newBoard[vac[0]][vac[1]] = side
            newVal = _alpha_beta(newState, otherMove, alpha, beta, plyLeft - 1, time_limit)
            if (newVal[1] > alpha):
                alpha = newVal[1]
                move = vac
            if (alpha >= beta):
                n_cutoffs += 1
                break
            # print(time.perf_counter() - StartTime)
            if (time.perf_counter() - start_time > time_limit):
                break
        return [move, alpha]
    else:
        for vac in all_vacancy:
            newState = curr_state.copy()
            newBoard = newState.board
            newBoard[vac[0]][vac[1]] = side
            newVal = _alpha_beta(newState, otherMove, alpha, beta, plyLeft - 1, time_limit)
            if (newVal[1] < beta):
                beta = newVal[1]
                move = vac
            if (alpha >= beta):
                n_cutoffs += 1
                break
            # print(time.perf_counter() - StartTime)
            if (time.perf_counter() - start_time > time_limit):
                break
        return [move, beta]



def _change_player(player):
    if (player == 'B'):
        return 'W'
    else:
        return 'B'

def take_turn(current_state, last_utterance, time_limit):
    # Compute the new state for a move.
    # Start by copying the current state.
    global start_time, total_move

    new_state = MY_TTS_State(current_state.board)
    # l = parameterized_minimax(current_state,
    #     max_ply = 2,
    #     alpha_beta = False,
    #     use_custom_static_eval_function = False,
    #     timed=False,
    #     time_limit=1.0)

    # best_val = l['CURRENT_STATE_STATIC_VAL']
    # new_state = mini_max_map[best_val]
    # Fix up whose turn it will be.
    who = current_state.whose_turn
    new_who = 'B'
    if who == 'B':
        new_who = 'W'
    new_state.whose_turn = new_who

    start_time = time.perf_counter()

    location = IDDFS(new_state, new_who, 10 ** 9, time_limit)

    new_state.board[location[0]][location[1]] = who

    # Construct a representation of the move that goes from the
    # currentState to the newState.
    move = location

    # Make up a new remark
    new_utterance = utterances[total_move % 12]
    total_move += 1

    return [[move, new_state], new_utterance]


def IDDFS(state, side, maxPly, time_limit):
    global start_time
    for ply in range(maxPly):
        if(Alpha_beta):
            result = _alpha_beta(state, side, -1000000, 1000000, ply, time_limit)
        else:
            result = _minimax(state, side, ply, time_limit)
        location = result[0]
        if (time.perf_counter() - start_time) > time_limit:
            return location

def _find_next_vacancy(b):
    for i in range(len(b)):
        for j in range(len(b[0])):
            if b[i][j] == ' ': return (i, j)
    return False

def _find_all_vacancy(b):
    all_vacancy = []
    for i in range(len(b)):
        for j in range(len(b[0])):
            if b[i][j] == ' ':
                all_vacancy.append((i, j))
    return all_vacancy


def moniker():
    return "McGeee"  # Return your agent's short nickname here.


def who_am_i():
    return """My name is My Man McGeee, created by Wayne Wang. I consider 
    myself to be an the best defender!."""


def get_ready(initial_state, k, who_i_play, player2Nickname):
    # do any prep, like eval pre-calculation, here.
    global My_Side
    global Board_Rows
    global Board_Cols
    global K
    global Opponent

    My_Side = who_i_play
    Board_Rows = len(initial_state.board)
    Board_Cols = len(initial_state.board[0])
    K = k
    Opponent = player2Nickname
    return "OK"
