from collections import defaultdict
import numpy as np

class UCT:
    """Implementation of UCT based on Leslie's lecture notes
    """
    def __init__(self, actions, reward_fn, transition_fn, done_fn=None, num_search_iters=100, gamma=0.9, seed=0):
        self._actions = actions
        self._reward_fn = reward_fn
        self._transition_fn = transition_fn
        self._done_fn = done_fn or (lambda s,a : False)
        self._num_search_iters = num_search_iters
        self._gamma = gamma
        self._rng = np.random.RandomState(seed)
        self._Q = None
        self._N = None

    def run(self, state, horizon=100):
        # Initialize Q[s][a][d] -> float
        self._Q = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
        # Initialize N[s][a][d] -> int
        self._N = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
        # Loop search
        for it in range(self._num_search_iters):
            # Update Q
            self._search(state, 0, horizon=horizon)

    def get_action(self, state, t=0):
        # Return best action, break ties randomly
        return max(self._actions, key=lambda a : (self._Q[state][a][t], self._rng.uniform()))

    def _search(self, s, depth, horizon=100):
        # Base case
        if depth == horizon:
            return 0.
        # Select an action, balancing explore/exploit
        a = self._select_action(s, depth, horizon=horizon)
        # Create a child state
        next_state = self._transition_fn(s, a)
        # Get value estimate
        if self._done_fn(s, a):
            # Some environments terminate problems before the horizon
            q = self._reward_fn(s, a)
        else:
            q = self._reward_fn(s, a) + self._gamma * self._search(next_state, depth+1, horizon=horizon)
        # Update values and counts
        num_visits = self._N[s][a][depth] # before now
        # First visit to (s, a, depth)
        if num_visits == 0:
            self._Q[s][a][depth] = q
        # We've been here before
        else:
            # Running average
            self._Q[s][a][depth] = (num_visits / (num_visits + 1.)) * self._Q[s][a][depth] + \
                                   (1 / (num_visits + 1.)) * q
        # Update num visits
        self._N[s][a][depth] += 1
        return self._Q[s][a][depth]

    def _select_action(self, s, depth, horizon):
        # If there is any action where N(s, a, depth) == 0, try it first
        untried_actions = [a for a in self._actions if self._N[s][a][depth] == 0]
        if len(untried_actions) > 0:
            return self._rng.choice(untried_actions)
        # Otherwise, take an action to trade off exploration and exploitation
        N_s_d = sum(self._N[s][a][depth] for a in self._actions)
        best_action_score = -np.inf
        best_actions = []
        for a in self._actions:
            # explore_bonus = (np.log(N_s_d) / self._N[s][a][depth])**((horizon + depth) / (2*horizon + depth))
            # explore_bonus = (np.log(N_s_d) / self._N[s][a][depth])**(0.5)
            explore_bonus = 0
            score = self._Q[s][a][depth] + explore_bonus
            if score > best_action_score:
                best_action_score = score
                best_actions = [a]
            elif score == best_action_score:
                best_actions.append(a)
        return self._rng.choice(best_actions)


if __name__ == "__main__":
    import imageio
    import time
    import pickle
    from tqdm import tqdm
    from tabulate import tabulate
    from envs.robot_kitchen import RobotKitchenEnv

    DEBUG = False

    approaches = []
    durations = []
    env_steps_taken = []
    all_returns = []

    render = True
    max_num_steps = 41
    gamma = 0.99

    for replanning_interval in [1]: #[1, 5, 10, float("inf")]:
        for num_search_iters in [1000]: #[10, 100, 1000]:
            approaches.append(f'UCT [Iters={num_search_iters}, Replan={replanning_interval}]')

            start_time = time.time()

            env = RobotKitchenEnv()

            uct = UCT(env.get_all_actions(), env.compute_reward, env.compute_transition, done_fn=env.compute_done,
                      num_search_iters=num_search_iters, gamma=gamma, seed=0)

            state, _ = env.reset()
            returns = 0.

            if render:
                images = []
                images.append(env.render())

            if DEBUG: print("Initial state:", env.state_to_str(state))
            for t in tqdm(range(max_num_steps)):
                if t % replanning_interval == 0:
                    if DEBUG: print("Running UCT...")
                    uct.run(state, horizon=max_num_steps-t)
                    steps_since_replanning = 0
                    if DEBUG: print("Done.")
                action = uct.get_action(state, t=steps_since_replanning)
                steps_since_replanning += 1
                if DEBUG: print("Taking action", action)
                state, reward, done, _ = env.step(action)
                returns += reward
                if DEBUG: print("State:", env.state_to_str(state))
                if DEBUG: print("Reward, Done:", reward, done)
                if render:
                    images.append(env.render())
                if done:
                    break

            if render:
                outfile = str(replanning_interval)+' '+str(num_search_iters)+" uct.mp4"
                imageio.mimsave(outfile, images)
                print("Wrote out to", outfile)

            duration = time.time() - start_time
            durations.append(duration)
            env_steps_taken.append(t)
            all_returns.append(returns)

    columns = ["Approach", "Duration (s)", "# Env Steps Taken", "Returns"]
    table = list(zip(approaches, durations, env_steps_taken, all_returns))
    print(tabulate(table, headers=columns))

    with open("uct_results.p", "wb") as f:
        pickle.dump(table, f)



