import agentpy as ap
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import numpy as np




# Collector Agent
class Collector(ap.Agent):
    def setup(self):
        self.grid = self.model.grid
    
    def move(self):
        dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        self.grid.move_by(self, (dx, dy))

# Waste collection model
class WasteModel(ap.Model):
    def setup(self):
        # Grid setup
        s = self.p.size
        n = self.p.agents
        self.grid = ap.Grid(self, (s, s), track_empty=True)

        # Create and add agents
        self.agents = ap.AgentList(self, n, Collector)
        self.grid.add_agents(self.agents, random=True, empty=True)

        # Save start positions for all agents
        self.start_positions = {}
        for agent in self.agents:
            self.start_positions[agent.id] = self.grid.positions[agent]
        self.mid_positions = {}
    
    # To do each step
    def step(self):
        self.agents.move()
        self.grid.record_positions(label="pos")
    
    # Record

    def end(self):
        # Save end positions for all agents
        self.end_positions = {}
        for agent in self.agents:
            self.end_positions[agent.id] = self.grid.positions[agent]

        # Print results
        for agent in self.agents:
            print(f"\n Agent {agent.id}: Start {self.start_positions[agent.id]} → End {self.end_positions[agent.id]}")

# Model run setup
parameters = {'agents': 3, 'steps': 20, 'size': 10}
model = WasteModel(parameters)
results = model.run()

# Getting positions for animation
df = results.variables.Collector
agent_positions = {}
for agent_id in df.index.get_level_values('obj_id').unique():
    positions = df.xs(agent_id, level='obj_id')[['pos0', 'pos1']].to_numpy()
    start_pos = model.start_positions[agent_id] 
    ps = np.vstack([start_pos, positions])
    agent_positions[agent_id] = ps
num_steps = next(iter(agent_positions.values())).shape[0]  
positions_per_step = []
for t in range(num_steps):
    step_positions = []
    for agent_id in agent_positions:
        step_positions.append(agent_positions[agent_id][t])
    positions_per_step.append(np.array(step_positions))

# ----- Animation ---------------------------------------------------------------------
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  

ax.set_xlim(-0.5, parameters['size']-0.5)
ax.set_ylim(-0.5, parameters['size']-0.5)
ax.set_xticks(range(parameters['size']))
ax.set_yticks(range(parameters['size']))
ax.set_aspect("equal")
ax.grid(True)

scat = ax.scatter([], [], s=100, c='red')
scat.set_offsets(positions_per_step[0])

# Slider for steps
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, '', 0, num_steps-1, valinit=0, valstep=1)

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
    step = max(step-1, 0)
    slider.set_val(step)

def next_step(event):
    step = int(slider.val)
    step = min(step+1, num_steps-1)
    slider.set_val(step)

btn_prev.on_clicked(prev_step)
btn_next.on_clicked(next_step)

plt.show()


