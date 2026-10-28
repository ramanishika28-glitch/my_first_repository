import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = 1.0          
A0 = 0.5         
NUM_STARS = 100  
DT = 0.1         

class GalaxySimulation:
    def __init__(self, num_stars):
        self.num_stars = num_stars
        self.pos = np.zeros((num_stars, 2))
        self.vel = np.zeros((num_stars, 2))
        self.core_mass = 5000.0
        self._initialize_galaxy()

    def _initialize_galaxy(self):
        for i in range(self.num_stars):
            radius = np.random.uniform(20, 150)
            angle = np.random.uniform(0, 2 * np.pi)
            self.pos[i, 0] = radius * np.cos(angle)
            self.pos[i, 1] = radius * np.sin(angle)
            v_mag = np.sqrt((G * self.core_mass) / radius)
            self.vel[i, 0] = -v_mag * np.sin(angle)
            self.vel[i, 1] = v_mag * np.cos(angle)

    def compute_accelerations(self, alternate_gravity=False):
        accel = np.zeros((self.num_stars, 2))
        for i in range(self.num_stars):
            r_vec = -self.pos[i]
            r_mag = np.linalg.norm(r_vec)
            if r_mag < 1.0: continue
            a_newtonian = (G * self.core_mass) / (r_mag ** 2)
            if alternate_gravity:
                a_mag = np.sqrt(a_newtonian * A0) if a_newtonian < A0 else a_newtonian
            else:
                a_mag = a_newtonian
            accel[i] = a_mag * (r_vec / r_mag)
        return accel

    def update(self, alternate_gravity=False):
        self.vel += self.compute_accelerations(alternate_gravity) * DT
        self.pos += self.vel * DT

sim = GalaxySimulation(NUM_STARS)
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
stars_plot = ax.scatter([], [], color='#00d2ff', s=4, alpha=0.8)
ax.scatter([0], [0], color='#ff3333', s=40, edgecolors='white')
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.axis('off')

USE_ALTERNATE_GRAVITY = True 
ax.set_title("Alternate Gravity Galaxy Simulation", color='white')

def init_anim():
    stars_plot.set_offsets(np.empty((0, 2)))
    return stars_plot,

def animate_frame(frame):
    sim.update(alternate_gravity=USE_ALTERNATE_GRAVITY)
    stars_plot.set_offsets(sim.pos)
    return stars_plot,

ani = animation.FuncAnimation(fig, animate_frame, init_func=init_anim, frames=200, interval=20, blit=True)
plt.show()
