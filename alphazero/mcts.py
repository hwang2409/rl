import math

import numpy as np

from game import Connect4
from model import AlphaZeroNet


class Node:
    __slots__ = ("state", "parent", "action", "children", "visit_count",
                 "value_sum", "prior", "is_expanded")

    def __init__(self, state: Connect4, parent: "Node | None" = None,
                 action: int | None = None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: dict[int, Node] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _ucb_score(parent: Node, child: Node, c_puct: float) -> float:
    """Upper confidence bound for tree selection."""
    exploration = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return child.q_value + exploration


def _select_child(node: Node, c_puct: float) -> tuple[int, Node]:
    """Select the child with highest UCB score."""
    best_score = -float("inf")
    best_action = -1
    best_child = None
    for action, child in node.children.items():
        score = _ucb_score(node, child, c_puct)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def _expand(node: Node, policy: np.ndarray) -> None:
    """Create child nodes for all valid moves using the neural net policy."""
    valid = node.state.get_valid_moves()

    # Mask invalid moves and renormalize
    policy = policy * valid
    policy_sum = policy.sum()
    if policy_sum > 0:
        policy = policy / policy_sum
    else:
        # Fallback: uniform over valid moves
        policy = valid.astype(np.float32) / valid.sum()

    for action in range(len(valid)):
        if valid[action]:
            child_state = node.state.make_move(action)
            node.children[action] = Node(
                state=child_state,
                parent=node,
                action=action,
                prior=policy[action],
            )
    node.is_expanded = True


def _backpropagate(node: Node, value: float) -> None:
    """Propagate the value up the tree, negating at each level.

    The value is from the perspective of node.state.current_player's OPPONENT
    (i.e., the player who just moved TO reach this node).
    As we go up, we negate because parent's current player is the opposite.
    """
    while node is not None:
        node.visit_count += 1
        # value is from the perspective of the player who moved to reach this node
        # which is the opponent of node.state.current_player
        node.value_sum += value
        value = -value  # flip perspective for parent
        node = node.parent


def search(state: Connect4, model: AlphaZeroNet, num_simulations: int,
           c_puct: float = 1.5, dirichlet_alpha: float = 1.0,
           dirichlet_epsilon: float = 0.25, add_noise: bool = True,
           device: str = "cpu") -> tuple[np.ndarray, float]:
    """Run MCTS from the given state.

    Returns:
        action_probs: visit count distribution over actions, shape (cols,)
        root_value: estimated value of the root position
    """
    root = Node(state)

    # Expand root with neural net
    policy, value = model.predict(state, device=device)
    _expand(root, policy)

    # Add Dirichlet noise to root for exploration
    if add_noise:
        valid_actions = list(root.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(valid_actions))
        for i, action in enumerate(valid_actions):
            child = root.children[action]
            child.prior = (1 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * noise[i]

    for _ in range(num_simulations):
        node = root

        # SELECT: walk down tree until we find an unexpanded node or terminal
        while node.is_expanded and node.children:
            _, node = _select_child(node, c_puct)

        # Check if terminal
        terminal, terminal_reward = node.state.is_terminal()
        if terminal:
            # terminal_reward is from the perspective of the player who just moved
            # (the parent's current_player), which is node.state.current_player's opponent
            _backpropagate(node, terminal_reward)
            continue

        # EXPAND + EVALUATE
        policy, value = model.predict(node.state, device=device)
        _expand(node, policy)

        # value is from current player's perspective at this node
        # backpropagate expects value from the perspective of the player who moved here
        # (the opponent of node.state.current_player), so we negate
        _backpropagate(node, -value)

    # Extract action probabilities from visit counts
    cols = state.cols
    action_probs = np.zeros(cols, dtype=np.float32)
    for action, child in root.children.items():
        action_probs[action] = child.visit_count

    # Normalize
    total = action_probs.sum()
    if total > 0:
        action_probs /= total

    root_value = root.q_value
    return action_probs, root_value


def select_action(action_probs: np.ndarray, temperature: float) -> int:
    """Select an action from the MCTS policy.

    temperature=1.0: sample proportionally to visit counts
    temperature~=0: pick the most visited action (argmax)
    """
    if temperature < 0.01:
        return int(np.argmax(action_probs))

    # Apply temperature
    probs = action_probs ** (1.0 / temperature)
    probs_sum = probs.sum()
    if probs_sum > 0:
        probs /= probs_sum
    else:
        # Fallback: uniform over non-zero original probs
        mask = action_probs > 0
        if mask.any():
            probs = mask.astype(np.float64) / mask.sum()
        else:
            probs = np.ones_like(action_probs, dtype=np.float64) / len(action_probs)
    # Fix floating point precision for np.random.choice
    probs = np.float64(probs)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
