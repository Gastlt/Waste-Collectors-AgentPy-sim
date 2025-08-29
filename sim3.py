import agentpy as ap
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from collections import defaultdict

# ------------------ Config ------------------
ACTIONS = [
    "MoveN","MoveS","MoveE","MoveW",
    "MoveNE","MoveNW","MoveSE","MoveSW",
    "Pickup","Drop","Refuel","Wait"
]
MOVE_VECTORS = {
    "MoveN": (0,-1), "MoveS": (0,1), "MoveE": (1,0), "MoveW": (-1,0),
    "MoveNE": (1,-1), "MoveNW": (-1,-1), "MoveSE": (1,1), "MoveSW": (-1,1),
}

# ------------------ Agent ------------------
class Collector(ap.Agent):
    def setup(self):
        self.grid = self.model.grid
        self.capacity = 0
        self.max_capacity = self.p.max_capacity
        self.fuel = self.p.initial_fuel
        self.state = "Idle"
        self.total_reward = 0.0

        # Q-learning
        self.q = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=float))
        self.epsilon = self.p.epsilon
        self.alpha   = self.p.alpha
        self.gamma   = self.p.gamma

    # ----- helpers -----
    def pos(self):
        return self.grid.positions[self]

    def encode_state(self):
        x, y = self.pos()
        load_bucket = int(self.capacity > 0)
        fuel_bucket = 0 if self.fuel > self.p.initial_fuel*0.5 else (1 if self.fuel > self.p.initial_fuel*0.2 else 2)
        on_bin = int(self.on_bin())       # 1 if standing on bin
        on_depot = int(self.on_depot())   # 1 if standing on depot
        return (x, y, load_bucket, fuel_bucket, on_bin, on_depot)


    def choose_action(self, s):
        if random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))
        return int(np.argmax(self.q[s]))

    def update_q(self, s, a, r, s2):
        best_next = np.max(self.q[s2])
        self.q[s][a] += self.alpha * (r + self.gamma * best_next - self.q[s][a])

    def on_depot(self):
        return self.pos() == self.model.depot

    def on_bin(self):
        return self.pos() in self.model.bins and self.model.bins[self.pos()] > 0

    @staticmethod
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    # ----- environment shaping targets -----
    def current_target(self):
        """If carrying capacity -> go depot; else nearest non-empty bin."""
        if self.capacity >= self.max_capacity or not self.model.has_bins_left():
            return self.model.depot
        # nearest non-empty bin
        myp = self.pos()
        candidates = [p for p, left in self.model.bins.items() if left > 0]
        if not candidates: return self.model.depot
        return min(candidates, key=lambda p: self.manhattan(myp, p))

    # ----- execute one action and return reward -----
    def take_action(self, action_idx):
        action = ACTIONS[action_idx]
        reward = 0.0

        # small per-step time penalty
        reward -= 0.01

        # movement
        if action in MOVE_VECTORS:
            old_p = self.pos()
            old_dist = self.manhattan(old_p, self.current_target())
            dx, dy = MOVE_VECTORS[action]

            # attempt bounded move; if out of bounds -> penalty
            x, y = old_p
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.p.size and 0 <= ny < self.p.size:
                self.grid.move_by(self, (dx, dy))
                # fuel cost (meters driven)
                reward -= 0.1
                self.state = "Navigating"

                # shaping: closer to target +1
                new_dist = self.manhattan((nx, ny), self.current_target())
                if new_dist < old_dist:
                    reward += 1.0
            else:
                # collision with boundary
                reward -= 6.0
                self.state = "Failure"

        # non-movement
        elif action == "Pickup":
            if self.on_bin() and self.capacity < self.max_capacity:
                self.capacity += 1
                self.model.bins[self.pos()] -= 1
                reward += 200.0
                self.state = "Servicing"
                print(f"Agent {self.id} picked up trash at {self.pos()}")

        elif action == "Drop":
            if self.on_depot():
                if self.capacity > 0:
                    reward += 15.0
                self.capacity = 0
                self.state = "Unloading"
            else:
                reward -= 1.0

        elif action == "Refuel":
            if self.on_depot():
                self.fuel = self.p.initial_fuel
                self.state = "Idle"
            else:
                reward -= 2.0

        elif action == "Wait":
            self.state = "Idle"

        # fuel tick + fail if empty
        self.fuel -= 1
        if self.fuel <= 0:
            reward -= 10.0
            self.state = "Failure"

        self.total_reward += reward
        return reward

# ------------------ Model ------------------
class WasteModel(ap.Model):
    def setup(self):
        s = self.p.size
        n = self.p.agents
        self.grid = ap.Grid(self, (s, s), track_empty=True)

        # depot at center
        self.depot = (s // 2, s // 2)

        # place bins (positions -> remaining units)
        rng = self.random
        self.bins = {}
        placed = 0
        while placed < self.p.n_bins:
            p = (rng.randint(0, s-1), rng.randint(0, s-1))
            if p != self.depot and p not in self.bins:
                self.bins[p] = self.p.bin_size
                placed += 1

        # agents
        self.agents = ap.AgentList(self, n, Collector)
        self.grid.add_agents(self.agents, random=True, empty=True)

        # start positions
        self.start_positions = {a.id: self.grid.positions[a] for a in self.agents}

        # tracking epsilon decay (optional)
        self.epsilon_decay = self.p.epsilon_decay

    def has_bins_left(self):
        return any(v > 0 for v in self.bins.values())

    def step(self):
        # one Q-learning step per agent
        for a in self.agents:
            s = a.encode_state()
            ai = a.choose_action(s)
            r = a.take_action(ai)
            s2 = a.encode_state()
            a.update_q(s, ai, r, s2)

        # record positions for plotting
        self.grid.record_positions(label="pos")

        if not self.has_bins_left() and all(a.capacity == 0 for a in self.agents):
            self.stop()

        # decay epsilon a bit each step
        for a in self.agents:
            a.epsilon = max(self.p.epsilon_min, a.epsilon * self.epsilon_decay)

    def update(self):
         for a in self.agents: # Records variables for each agent
            self.record(f'fuel_{a.id}', a.fuel)
            self.record(f'cap_{a.id}', a.capacity)
            self.record(f'reward_{a.id}', a.total_reward)
            self.record(f'state_{a.id}', a.state)

    def end(self):
        self.end_positions = {a.id: self.grid.positions[a] for a in self.agents}
        for a in self.agents:
            print(f"Agent {a.id}: Start {self.start_positions[a.id]} → End {self.end_positions[a.id]} | "
                  f"Total reward={a.total_reward:.2f}, fuel={a.fuel}, cap={a.capacity}")

# ------------------ Run once ------------------
parameters = {
    'size': 12,
    'agents': 3,
    'steps': 200,           # upper bound; model may stop earlier
    'n_bins': 5,
    'bin_size': 4,          # how many pickups per bin until empty
    'max_capacity': 6,
    'initial_fuel': 200,

    # Q-learning
    'epsilon': 0.25,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.995,
    'alpha': 0.2,
    'gamma': 0.95,
}

model = WasteModel(parameters)
results = model.run()

# ------------------ Build positions for your slider UI ------------------
# AgentPy stores per-agent positions in results.variables.<AgentClass>
df = results.variables.Collector  # MultiIndex: (obj_id, t) -> pos0, pos1
agent_positions = {}
for agent_id in df.index.get_level_values('obj_id').unique():
    # rows for this agent, ordered by time
    arr = df.xs(agent_id, level='obj_id')[['pos0', 'pos1']].to_numpy()
    # prepend start pos (t=0)
    start_pos = model.start_positions[agent_id]
    arr = np.vstack([np.array(start_pos, dtype=float), arr])
    agent_positions[agent_id] = arr

num_steps = next(iter(agent_positions.values())).shape[0]

positions_per_step = []
for t in range(num_steps):
    step_positions = []
    for aid in sorted(agent_positions.keys()):
        step_positions.append(agent_positions[aid][t])
    positions_per_step.append(np.array(step_positions))

# ------------------ Matplotlib step-by-step UI (slider + prev/next) ------------------
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

ax.set_xlim(-0.5, parameters['size'] - 0.5)
ax.set_ylim(-0.5, parameters['size'] - 0.5)
ax.set_xticks(range(parameters['size']))
ax.set_yticks(range(parameters['size']))
ax.set_aspect("equal")
ax.grid(True)

# draw depot + bins
ax.scatter([model.depot[0]], [model.depot[1]], s=180, marker='s', label='Depot')
if model.bins:
    bx, by = zip(*model.bins.keys())
    ax.scatter(bx, by, s=120, marker='P', label='Bins')

scat = ax.scatter([], [], s=100, c='red', label='Agents')
scat.set_offsets(positions_per_step[0])
ax.legend(loc='upper right')

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Step', 0, num_steps-1, valinit=0, valstep=1)

def update_slider(val):
    step = int(slider.val)
    scat.set_offsets(positions_per_step[step])
    fig.canvas.draw_idle()

slider.on_changed(update_slider)

# Prev/Next buttons
ax_prev = plt.axes([0.05, 0.1, 0.1, 0.04])
ax_next = plt.axes([0.85, 0.1, 0.1, 0.04])
btn_prev = Button(ax_prev, 'Prev')
btn_next = Button(ax_next, 'Next')

def prev_step(event):
    step = int(slider.val)
    slider.set_val(max(step-1, 0))

def next_step(event):
    step = int(slider.val)
    step = min(step+1, num_steps-1)
    slider.set_val(step)

    print(f"\n=== Step {step} ===")
    for agent, pos in zip(model.agents, positions_per_step[step]):
        px, py = map(int, pos)

        # fetch historical variables from results
        fuel = results.variables.WasteModel[f'fuel_{agent.id}'].iloc[step]
        cap = results.variables.WasteModel[f'cap_{agent.id}'].iloc[step]
        reward = results.variables.WasteModel[f'reward_{agent.id}'].iloc[step]
        state = results.variables.WasteModel[f'state_{agent.id}'].iloc[step]

        print(f"Agent {agent.id}: pos=({px},{py}), fuel={fuel}, cap={cap}, reward={reward:.2f}, state={state}")


btn_prev.on_clicked(prev_step)
btn_next.on_clicked(next_step)

plt.show()
