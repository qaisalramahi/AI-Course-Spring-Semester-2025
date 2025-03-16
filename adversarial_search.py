import math
import pygame
import numpy as np

# Import constants directly from the provided environment.py
from environment import (
    PATH_TILES, GAS_TILES, GAS_MAX, GOAL_POS, START_POS,
    GRID_SIZE, MAP_SPRITE, CAR_SPRITE, TROPHY_SPRITE, CELL_SIZE
)

# --- Helper functions for move simulation ---

def get_possible_actions_car(car_pos, blocker_pos):
    """
    Return legal actions (0=Left, 1=Right, 2=Up, 3=Down) for the car,
    based on the car's current position and excluding moves that would 
    lead to the cell occupied by the blocker.
    """
    actions = []
    x, y = car_pos
    # Check left
    if (x - 1, y) in PATH_TILES and (x - 1, y) != blocker_pos:
        actions.append(0)
    # Check right
    if (x + 1, y) in PATH_TILES and (x + 1, y) != blocker_pos:
        actions.append(1)
    # Check up
    if (x, y - 1) in PATH_TILES and (x, y - 1) != blocker_pos:
        actions.append(2)
    # Check down
    if (x, y + 1) in PATH_TILES and (x, y + 1) != blocker_pos:
        actions.append(3)
    
    return actions

def get_possible_actions_blocker(blocker_pos):
    """
    For the blocker, allow movement in the four cardinal directions if within grid boundaries.
    The blocker is not restricted to PATH_TILES.
    """
    actions = []
    x, y = blocker_pos
    grid_width, grid_height = GRID_SIZE[1], GRID_SIZE[0]  # GRID_SIZE is (rows, cols)
    if x - 1 >= 0:
        actions.append(0)
    if x + 1 < grid_width:
        actions.append(1)
    if y - 1 >= 0:
        actions.append(2)
    if y + 1 < grid_height:
        actions.append(3)
    return actions

def simulate_car_move(car_pos, action):
    """
    Simulate the car's move given an action.
    Returns (new_position, immediate_reward). If move is invalid, return (None, penalty).
    """
    x, y = car_pos
    if action == 0:  # Left
        new_pos = (x - 1, y)
    elif action == 1:  # Right
        new_pos = (x + 1, y)
    elif action == 2:  # Up
        new_pos = (x, y - 1)
    elif action == 3:  # Down
        new_pos = (x, y + 1)
    else:
        new_pos = car_pos

    # Check if the new cell is a valid path tile
    if new_pos not in PATH_TILES:
        return None, -10.0  # Same penalty as in environment.py

    # Basic move penalty (same as in environment.py)
    move_reward = -0.1

    # Add bonus reward for getting closer to the goal
    old_dist = abs(car_pos[0] - GOAL_POS[0]) + abs(car_pos[1] - GOAL_POS[1])
    new_dist = abs(new_pos[0] - GOAL_POS[0]) + abs(new_pos[1] - GOAL_POS[1])
    if new_dist < old_dist:
        move_reward += 2.0  # Bonus for moving closer to goal

    # If the car reaches the goal, add a large reward (same as in environment.py)
    if new_pos == GOAL_POS:
        move_reward += 100.0

    return new_pos, move_reward

def simulate_blocker_move(blocker_pos, action):
    """
    Simulate the blocker's move given an action.
    Returns the new blocker position.
    """
    x, y = blocker_pos
    if action == 0:  # Left
        new_pos = (x - 1, y)
    elif action == 1:  # Right
        new_pos = (x + 1, y)
    elif action == 2:  # Up
        new_pos = (x, y - 1)
    elif action == 3:  # Down
        new_pos = (x, y + 1)
    else:
        new_pos = blocker_pos
    
    # Ensure blocker stays within grid bounds
    x, y = new_pos
    x = max(0, min(x, GRID_SIZE[0] - 1))
    y = max(0, min(y, GRID_SIZE[1] - 1))
    return (x, y)

# --- Evaluation function and minimax search ---

def evaluate_state(car_pos, blocker_pos, car_path=None):
    """
    Heuristic evaluation of the state.
    Returns a high positive value if the car reaches the goal and a high negative value if
    the car collides with the blocker.
    Otherwise, it rewards states with a lower Manhattan distance to the goal
    and penalizes when the blocker is too near.
    """
    if car_pos == GOAL_POS:
        return 10000  # Winning state
    if car_pos == blocker_pos:
        return -5000  # Collision penalty

    # Distance-based heuristic
    dist_to_goal = abs(car_pos[0] - GOAL_POS[0]) + abs(car_pos[1] - GOAL_POS[1])
    goal_component = -dist_to_goal * 30  # Heavy weight on goal distance
    
    # Blocker proximity penalty
    dist_blocker = abs(car_pos[0] - blocker_pos[0]) + abs(car_pos[1] - blocker_pos[1])
    blocker_penalty = 0
    if dist_blocker < 4:  # Increased safety distance
        blocker_penalty = (4 - dist_blocker) * 50
    
    # Additional heuristic: Check if blocker is between car and goal
    # This helps the car avoid situations where the blocker can trap it
    between_penalty = 0
    
    # If car and goal are aligned horizontally
    if car_pos[1] == GOAL_POS[1]:
        min_x = min(car_pos[0], GOAL_POS[0])
        max_x = max(car_pos[0], GOAL_POS[0])
        if blocker_pos[1] == car_pos[1] and min_x < blocker_pos[0] < max_x:
            between_penalty = 100
    
    # If car and goal are aligned vertically
    if car_pos[0] == GOAL_POS[0]:
        min_y = min(car_pos[1], GOAL_POS[1])
        max_y = max(car_pos[1], GOAL_POS[1])
        if blocker_pos[0] == car_pos[0] and min_y < blocker_pos[1] < max_y:
            between_penalty = 100
    
    # Path quality: incentivize having move options
    path_quality = len(get_possible_actions_car(car_pos, blocker_pos)) * 10
    
    return goal_component - blocker_penalty - between_penalty + path_quality

def minimax(state, depth, alpha, beta, max_depth):
    """
    Perform minimax search with alpha-beta pruning.
    state is a tuple: (car_pos, blocker_pos, turn)
      - turn: "car" or "blocker"
    Returns (value, best_action).
    """
    car_pos, blocker_pos, turn = state
    original_depth = max_depth - depth

    # Terminal conditions: depth reached or terminal state encountered
    if depth == 0 or car_pos == GOAL_POS or car_pos == blocker_pos:
        eval_value = evaluate_state(car_pos, blocker_pos)
        # Apply depth discount to favor earlier goal achievement
        if car_pos == GOAL_POS:
            eval_value -= original_depth * 10
        return eval_value, None

    if turn == "car":
        best_value = -math.inf
        best_action = None
        actions = get_possible_actions_car(car_pos, blocker_pos)
        if not actions:
            return evaluate_state(car_pos, blocker_pos), None
        
        # Sort actions by preliminary evaluation to try more promising moves first (pruning optimization)
        action_scores = []
        for action in actions:
            new_car_pos, _ = simulate_car_move(car_pos, action)
            if new_car_pos is None:
                continue
            # Simple heuristic: prefer moves that get closer to the goal
            dist_to_goal = abs(new_car_pos[0] - GOAL_POS[0]) + abs(new_car_pos[1] - GOAL_POS[1])
            action_scores.append((action, -dist_to_goal))
        
        # Sort actions by score (descending)
        sorted_actions = [a for a, s in sorted(action_scores, key=lambda x: x[1], reverse=True)]
        
        for action in sorted_actions:
            new_car_pos, move_reward = simulate_car_move(car_pos, action)
            if new_car_pos is None:
                continue  # skip invalid move
            # After car moves, it's the blocker's turn.
            new_state = (new_car_pos, blocker_pos, "blocker")
            value, _ = minimax(new_state, depth - 1, alpha, beta, max_depth)
            value += move_reward  # include immediate reward
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break  # beta cutoff
        return best_value, best_action
    else:
        # Blocker turn (minimizing player)
        best_value = math.inf
        best_action = None
        actions = get_possible_actions_blocker(blocker_pos)
        if not actions:
            return evaluate_state(car_pos, blocker_pos), None
        
        # Sort actions by preliminary evaluation to try more promising moves first (pruning optimization)
        action_scores = []
        for action in actions:
            new_blocker_pos = simulate_blocker_move(blocker_pos, action)
            # Simple heuristic: prefer moves that get closer to the car
            dist_to_car = abs(new_blocker_pos[0] - car_pos[0]) + abs(new_blocker_pos[1] - car_pos[1])
            action_scores.append((action, dist_to_car))
        
        # Sort actions by score (ascending for blocker - it wants to minimize)
        sorted_actions = [a for a, s in sorted(action_scores, key=lambda x: x[1])]
        
        for action in sorted_actions:
            new_blocker_pos = simulate_blocker_move(blocker_pos, action)
            new_state = (car_pos, new_blocker_pos, "car")
            value, _ = minimax(new_state, depth - 1, alpha, beta, max_depth)
            if value < best_value:
                best_value = value
                best_action = action
            beta = min(beta, best_value)
            if beta <= alpha:
                break  # alpha cutoff
        return best_value, best_action

def minimax_decision(state, depth):
    """
    Returns the best action for the current player using minimax search.
    """
    _, action = minimax(state, depth, -math.inf, math.inf, depth)
    return action

def action_to_str(action):
    mapping = {0: "Left", 1: "Right", 2: "Up", 3: "Down"}
    return mapping.get(action, "Stay")

# --- Rendering functions ---

def render_state(car_pos, blocker_pos, screen, prev_car_pos=None):
    """
    Renders the current state: map background, goal, car, and blocker.
    """
    # Clear the screen.
    screen.fill((0, 0, 0))
    
    # Determine window size.
    width = GRID_SIZE[1] * CELL_SIZE
    height = GRID_SIZE[0] * CELL_SIZE
    
    # Draw the map background.
    scaled_background = pygame.transform.scale(MAP_SPRITE, (width, height))
    screen.blit(scaled_background, (0, 0))
    
    # Highlight PATH_TILES with a slightly visible overlay for debugging
    for tile in PATH_TILES:
        x, y = tile
        tile_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (100, 100, 255, 50), tile_rect, 1)
    
    # Highlight GAS_TILES
    for tile in GAS_TILES:
        x, y = tile
        tile_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (0, 255, 0, 100), tile_rect, 2)
    
    # Draw the goal trophy.
    goal_x, goal_y = GOAL_POS
    trophy_rect = pygame.Rect(goal_x * CELL_SIZE, goal_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    scaled_trophy = pygame.transform.scale(TROPHY_SPRITE, (CELL_SIZE, CELL_SIZE))
    screen.blit(scaled_trophy, trophy_rect)
    
    # Draw the car.
    car_x, car_y = car_pos
    car_rect = pygame.Rect(car_x * CELL_SIZE, car_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    scaled_car = pygame.transform.scale(CAR_SPRITE, (CELL_SIZE, CELL_SIZE))
    
    # Rotate car based on movement direction if prev_pos is available
    if prev_car_pos:
        if prev_car_pos[0] < car_pos[0]:  # Moving right
            scaled_car = pygame.transform.rotate(scaled_car, 90)
        elif prev_car_pos[0] > car_pos[0]:  # Moving left
            scaled_car = pygame.transform.rotate(scaled_car, -90)
        elif prev_car_pos[1] < car_pos[1]:  # Moving down
            scaled_car = pygame.transform.rotate(scaled_car, 180)
        # Moving up = default orientation
    
    screen.blit(scaled_car, car_rect)
    
    # Draw the blocker as a red rectangle.
    blocker_x, blocker_y = blocker_pos
    blocker_rect = pygame.Rect(blocker_x * CELL_SIZE, blocker_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, (255, 0, 0), blocker_rect)
    
    # Display distance to goal
    font = pygame.font.Font(None, 36)
    dist_to_goal = abs(car_pos[0] - GOAL_POS[0]) + abs(car_pos[1] - GOAL_POS[1])
    dist_text = font.render(f"Distance to Goal: {dist_to_goal}", True, (255, 255, 255))
    screen.blit(dist_text, (10, 10))
    
    # Display number of possible actions for car
    actions = get_possible_actions_car(car_pos, blocker_pos)
    actions_text = font.render(f"Possible Moves: {len(actions)}", True, (255, 255, 255))
    screen.blit(actions_text, (10, 50))
    
    pygame.display.flip()

# --- Main simulation loop demonstrating adversarial search with rendering ---

def main():
    pygame.init()
    width = GRID_SIZE[1] * CELL_SIZE
    height = GRID_SIZE[0] * CELL_SIZE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Adversarial Car vs. Blocker - Race to the Goal")
    
    # Initial state:
    # Car starts at START_POS.
    # Position the blocker where it won't immediately block the car
    car_pos = START_POS
    prev_car_pos = car_pos
    blocker_pos = (5, 5)  # Starting away from car's path
    state = (car_pos, blocker_pos, "car")

    # Higher search depth for better planning
    depth_limit = 7
    turn = "car"
    move_number = 0

    clock = pygame.time.Clock()
    
    # Keep track of visited positions to detect cycles
    car_visited = {car_pos: 1}
    cycle_detection_limit = 3  # Max visits to same position before considering it a cycle

    print("Starting adversarial simulation with a blocker obstacle.")
    running = True
    while running:
        # Process pygame events to allow window closure.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Render the current state.
        render_state(car_pos, blocker_pos, screen, prev_car_pos)
        pygame.time.wait(500)  # Pause for visualization
        
        print(f"\nMove {move_number}:")
        print(f"  Car: {car_pos}")
        print(f"  Blocker: {blocker_pos}")
        print(f"  Distance to goal: {abs(car_pos[0] - GOAL_POS[0]) + abs(car_pos[1] - GOAL_POS[1])}")

        # Terminal state checks.
        if car_pos == GOAL_POS:
            print("Car reached the goal!")
            font = pygame.font.Font(None, 72)
            win_text = font.render("GOAL REACHED!", True, (0, 255, 0))
            text_rect = win_text.get_rect(center=(width/2, height/2))
            screen.blit(win_text, text_rect)
            pygame.display.flip()
            pygame.time.wait(3000)  # Show winning message for 3 seconds
            running = False
            continue
        if car_pos == blocker_pos:
            print("Blocker collided with the car!")
            font = pygame.font.Font(None, 72)
            lose_text = font.render("Busted!", True, (255, 0, 0))
            text_rect = lose_text.get_rect(center=(width/2, height/2))
            screen.blit(lose_text, text_rect)
            pygame.display.flip()
            pygame.time.wait(3000)  # Show losing message for 3 seconds
            running = False
            continue
            
        # Check for cycles in car movement
        if car_pos in car_visited:
            car_visited[car_pos] += 1
            if car_visited[car_pos] > cycle_detection_limit:
                print(f"Warning: Car is cycling through position {car_pos}")
                # Randomly pick a different action to break the cycle if minimax is stuck
                if move_number % 5 == 0:  # Only intervene occasionally
                    actions = get_possible_actions_car(car_pos, blocker_pos)
                    if len(actions) > 1:
                        print("Breaking cycle with random move")
                        turn = "car"
                        continue

        if turn == "car":
            # For car, we may increase search depth when closer to goal
            dist_to_goal = abs(car_pos[0] - GOAL_POS[0]) + abs(car_pos[1] - GOAL_POS[1])
            local_depth = min(depth_limit + 3, depth_limit + 5 if dist_to_goal < 5 else depth_limit)
            print(f"  Car searching with depth {local_depth}")
            
            action = minimax_decision(state, local_depth)
            if action is None:
                print("No valid moves for the car. Simulation ends.")
                running = False
                continue
            print(f"Car chooses to move: {action_to_str(action)}")
            prev_car_pos = car_pos
            new_car_pos, reward = simulate_car_move(car_pos, action)
            if new_car_pos is None:
                print("Car made an invalid move. Simulation ends.")
                running = False
                continue
            car_pos = new_car_pos
            state = (car_pos, blocker_pos, "blocker")
            turn = "blocker"
        else:  # Blocker's turn
            # For blocker, use deeper search when near the car
            dist_to_car = abs(car_pos[0] - blocker_pos[0]) + abs(car_pos[1] - blocker_pos[1])
            local_depth = min(depth_limit + 1, depth_limit + 3 if dist_to_car < 3 else depth_limit)
            print(f"  Blocker searching with depth {local_depth}")
            
            action = minimax_decision(state, local_depth)
            if action is None:
                print("No valid moves for the blocker. Simulation ends.")
                running = False
                continue
            print(f"Blocker chooses to move: {action_to_str(action)}")
            blocker_pos = simulate_blocker_move(blocker_pos, action)
            state = (car_pos, blocker_pos, "car")
            turn = "car"

        move_number += 1
        if move_number > 150:  # Generous move limit
            print("Move limit reached. Ending simulation.")
            running = False

        clock.tick(15)  # Limit to 15 FPS

    pygame.quit()

if __name__ == "__main__":
    main()