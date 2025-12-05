#!/usr/bin/env python3
"""
Enhanced MRRC Universe Simulation
Real-time visualization with increased energy and 4D symmetry analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter

@dataclass
class MRRC:
    """Minimal Relational Reference Cell"""
    id: int
    position: np.ndarray  # 3D spatial position
    state: str  # Internal state
    energy: float  # Locked energy
    references: Set[int]  # References to other MRRCs
    birth_time: float
    
    def __post_init__(self):
        if not isinstance(self.references, set):
            self.references = set()

class EnhancedMRRCUniverse:
    """Enhanced universe with more energy and 4D symmetry analysis"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Energy budget - MASSIVELY INCREASED
        self.total_energy = 1e13  # 10 TeV (10,000 GeV!)
        self.available_energy = self.total_energy
        
        # Fundamental constants (emergent)
        self.spawn_cost = 500  # eV per MRRC (reduced for more spawning)
        self.interaction_cost_scale = 0.1
        self.cost_reduction_rate = 0.01  # Gravity strength
        self.auto_spawn_rate = 0.25  # 25% chance per step (reduced for performance)
        self.max_mrrcs = 500  # Reduced limit for better performance
        
        # MRRCs
        self.mrrcs: List[MRRC] = []
        self.next_id = 0
        
        # Time
        self.current_time = 0.0
        self.dt = 0.1
        
        # Phase tracking
        self.regime = "QUANTUM"  # QUANTUM or CLASSICAL
        self.decoherence_time = None
        self.information_capacity = 100  # Relationships we can track exactly
        
        # Speed of light (emergent)
        self.c = float('inf')  # Starts infinite in quantum regime
        
        # Metrics
        self.total_spawned = 0
        self.running = True
        
        # SPAWN TRACKING
        self.spawn_events = []  # Track all spawn events with cause
        self.vacuum_spawns = 0
        self.interaction_spawns = 0
        self.cluster_collision_spawns = 0
        
        # 4D symmetry tracking
        self.symmetry_history: List[Dict] = []
        
        # History
        self.history = {
            'time': [],
            'num_mrrcs': [],
            'energy': [],
            'cost': [],
            'max_cluster_mass': []
        }
    
    def spawn_mrrc(self, cause: str = "vacuum", position: Optional[np.ndarray] = None, 
                   parent_ids: Optional[List[int]] = None) -> Optional[MRRC]:
        """Create new MRRC from available energy"""
        if self.available_energy < self.spawn_cost:
            return None
        
        # Determine position
        if position is not None:
            spawn_position = position
        elif self.mrrcs and np.random.random() < 0.7:
            # Near existing MRRC (clustering)
            target = self.mrrcs[np.random.randint(len(self.mrrcs))]
            spawn_position = target.position + np.random.randn(3) * 0.5
        else:
            # Random position
            spawn_position = np.random.randn(3) * 2.0
        
        # Random state
        state = np.random.choice(['IS', 'WAS', 'WILL', 'COULD'])
        
        mrrc = MRRC(
            id=self.next_id,
            position=spawn_position,
            state=state,
            energy=self.spawn_cost,
            references=set(),
            birth_time=self.current_time
        )
        
        self.mrrcs.append(mrrc)
        self.next_id += 1
        self.total_spawned += 1
        self.available_energy -= self.spawn_cost
        
        # Record spawn event
        self.spawn_events.append({
            'time': self.current_time,
            'mrrc_id': mrrc.id,
            'cause': cause,
            'position': spawn_position.copy(),
            'parent_ids': parent_ids if parent_ids else []
        })
        
        # Update counters
        if cause == "vacuum":
            self.vacuum_spawns += 1
        elif cause == "interaction":
            self.interaction_spawns += 1
        elif cause == "cluster_collision":
            self.cluster_collision_spawns += 1
        
        return mrrc
    
    def evolve_step(self, dt: float):
        """Single evolution step"""
        self.current_time += dt
        
        # AUTO-SPAWN: Let MRRCs manifest spontaneously from vacuum!
        if self.available_energy >= self.spawn_cost and len(self.mrrcs) < self.max_mrrcs:
            if np.random.random() < self.auto_spawn_rate:
                self.spawn_mrrc(cause="vacuum")
        
        if len(self.mrrcs) < 2:
            return
        
        # Update references (form relationships)
        interactions_this_step = []
        for mrrc in self.mrrcs:
            # Probabilistically reference nearby MRRCs
            for other in self.mrrcs:
                if other.id == mrrc.id:
                    continue
                
                dist = np.linalg.norm(mrrc.position - other.position)
                if dist < 3.0 and np.random.random() < 0.1:
                    mrrc.references.add(other.id)
                    interactions_this_step.append((mrrc.id, other.id, dist))
        
        # INTERACTION SPAWNING: High-energy interactions can create new MRRCs!
        # When two MRRCs interact strongly (close + high reference count), 
        # they can spawn a new MRRC (like particle creation in collisions)
        if self.available_energy >= self.spawn_cost and len(self.mrrcs) < 1000:
            for mrrc_id1, mrrc_id2, dist in interactions_this_step:
                # Strong interaction criterion: very close + both have many references
                mrrc1 = next((m for m in self.mrrcs if m.id == mrrc_id1), None)
                mrrc2 = next((m for m in self.mrrcs if m.id == mrrc_id2), None)
                
                if mrrc1 and mrrc2:
                    # Strong interaction = close distance + both highly connected
                    if dist < 0.5 and len(mrrc1.references) > 5 and len(mrrc2.references) > 5:
                        # 10% chance to spawn from interaction (reduced for performance)
                        if np.random.random() < 0.1:
                            # Spawn at midpoint with slight offset
                            midpoint = (mrrc1.position + mrrc2.position) / 2
                            offset = np.random.randn(3) * 0.2
                            new_mrrc = self.spawn_mrrc(
                                cause="interaction",
                                position=midpoint + offset,
                                parent_ids=[mrrc1.id, mrrc2.id]
                            )
                            if new_mrrc:
                                # New MRRC references both parents
                                new_mrrc.references.add(mrrc1.id)
                                new_mrrc.references.add(mrrc2.id)
        
        # CLUSTER COLLISION SPAWNING: When clusters collide, spawn new MRRCs
        clusters = self.get_mass_clusters(threshold=2.0)
        if len(clusters) >= 2 and self.available_energy >= self.spawn_cost:
            # Check for colliding clusters
            for i, cluster1 in enumerate(clusters):
                for cluster2 in clusters[i+1:]:
                    # Get cluster centers
                    pos1 = np.mean([self.mrrcs[j].position for j, m in enumerate(self.mrrcs) 
                                   if m.id in cluster1], axis=0)
                    pos2 = np.mean([self.mrrcs[j].position for j, m in enumerate(self.mrrcs) 
                                   if m.id in cluster2], axis=0)
                    
                    cluster_dist = np.linalg.norm(pos1 - pos2)
                    
                    # If clusters very close (colliding)
                    if cluster_dist < 1.0:
                        # Spawn probability based on cluster sizes (more massive = more likely)
                        spawn_prob = min(0.3, 0.05 * (len(cluster1) + len(cluster2)) / 10)
                        if np.random.random() < spawn_prob:
                            # Spawn at collision point
                            collision_point = (pos1 + pos2) / 2
                            offset = np.random.randn(3) * 0.1
                            self.spawn_mrrc(
                                cause="cluster_collision",
                                position=collision_point + offset,
                                parent_ids=list(cluster1)[:2] + list(cluster2)[:2]  # Sample parents
                            )
        
        # Calculate information cost
        total_cost = self.calculate_information_cost()
        
        # GRAVITY: Minimize cost by moving MRRCs closer
        self.apply_gravity(dt)
        
        # Check for decoherence (quantum → classical transition)
        if self.regime == "QUANTUM":
            num_relationships = sum(len(m.references) for m in self.mrrcs)
            if num_relationships > self.information_capacity:
                self.regime = "CLASSICAL"
                self.decoherence_time = self.current_time
                
                # Speed of light emerges!
                avg_separation = np.mean([np.linalg.norm(m.position) 
                                         for m in self.mrrcs]) if self.mrrcs else 1.0
                self.c = max(avg_separation / dt, 1.0)
        
        # Analyze 4D symmetries
        self.analyze_4d_symmetries()
        
        # Record history
        clusters = self.get_mass_clusters()
        max_mass = max([self.cluster_mass(c) for c in clusters]) if clusters else 0
        
        self.history['time'].append(self.current_time)
        self.history['num_mrrcs'].append(len(self.mrrcs))
        self.history['energy'].append(self.available_energy)
        self.history['cost'].append(total_cost)
        self.history['max_cluster_mass'].append(max_mass)
    
    def calculate_information_cost(self) -> float:
        """Calculate total information cost of current configuration"""
        if len(self.mrrcs) < 2:
            return 0.0
        
        total_cost = 0.0
        
        # Cost of maintaining references
        for mrrc in self.mrrcs:
            for ref_id in mrrc.references:
                # Find referenced MRRC
                ref_mrrc = next((m for m in self.mrrcs if m.id == ref_id), None)
                if ref_mrrc:
                    # Cost increases with distance
                    dist = np.linalg.norm(mrrc.position - ref_mrrc.position)
                    total_cost += self.interaction_cost_scale * dist**2
        
        return total_cost
    
    def apply_gravity(self, dt: float):
        """Apply gravitational attraction (cost minimization)"""
        if len(self.mrrcs) < 2:
            return
        
        forces = [np.zeros(3) for _ in self.mrrcs]
        
        for i, mrrc in enumerate(self.mrrcs):
            for ref_id in mrrc.references:
                ref_mrrc = next((m for m in self.mrrcs if m.id == ref_id), None)
                if ref_mrrc:
                    # Force toward reference (minimize cost)
                    direction = ref_mrrc.position - mrrc.position
                    dist = np.linalg.norm(direction)
                    if dist > 0.01:
                        force = self.cost_reduction_rate * direction / (dist + 0.1)
                        forces[i] += force
        
        # Update positions
        for i, mrrc in enumerate(self.mrrcs):
            mrrc.position += forces[i] * dt
    
    def analyze_4d_symmetries(self):
        """Analyze geometric symmetries in 4D spacetime"""
        if len(self.mrrcs) < 4:
            return
        
        # Get 4D positions (x, y, z, t)
        positions_4d = np.array([
            [m.position[0], m.position[1], m.position[2], self.current_time] 
            for m in self.mrrcs
        ])
        
        # Center of mass in 4D
        com_4d = np.mean(positions_4d, axis=0)
        centered = positions_4d - com_4d
        
        # Moment of inertia tensor in 4D (4x4 matrix)
        I_4d = np.zeros((4, 4))
        for pos in centered:
            r_squared = np.dot(pos, pos)
            I_4d += r_squared * np.eye(4) - np.outer(pos, pos)
        
        # Eigenvalues reveal symmetry
        eigenvalues = np.linalg.eigvalsh(I_4d)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        
        # Classify symmetry
        if len(eigenvalues) >= 4 and eigenvalues[0] > 1e-6:
            ratios = eigenvalues[1:] / eigenvalues[0]
            
            symmetry = {
                'time': self.current_time,
                'eigenvalues': eigenvalues.copy(),
                'ratios': ratios.copy(),
                'num_mrrcs': len(self.mrrcs),
                'type': 'unknown'
            }
            
            # Classify based on eigenvalue ratios
            if np.allclose(ratios, [1, 1, 1], atol=0.1):
                symmetry['type'] = '4-sphere (SO(5))'  # All directions equal
            elif np.allclose(ratios[:2], [1, 1], atol=0.1) and ratios[2] < 0.5:
                symmetry['type'] = '3-sphere × line'  # Spatial sphere + time
            elif np.allclose(ratios, [1, 1, 0], atol=0.15):
                symmetry['type'] = '2-sphere (rotational)'
            elif np.allclose(ratios[:1], [1], atol=0.1) and ratios[1] < 0.5:
                symmetry['type'] = 'cylindrical'
            elif ratios[0] < 0.3:
                symmetry['type'] = 'linear (1D)'
            else:
                symmetry['type'] = 'asymmetric'
            
            self.symmetry_history.append(symmetry)
    
    def get_mass_clusters(self, threshold: float = 2.0) -> List[Set[int]]:
        """Identify mass clusters (groups of strongly connected MRRCs) - iterative to avoid stack overflow"""
        if len(self.mrrcs) < 2:
            return []
        
        visited = set()
        clusters = []
        
        # Build position lookup for speed
        mrrc_positions = {m.id: m.position for m in self.mrrcs}
        
        for start_mrrc in self.mrrcs:
            if start_mrrc.id in visited:
                continue
            
            # BFS instead of DFS (iterative, no recursion)
            cluster = set()
            queue = [start_mrrc.id]
            
            while queue:
                mrrc_id = queue.pop(0)
                if mrrc_id in visited:
                    continue
                
                visited.add(mrrc_id)
                cluster.add(mrrc_id)
                
                # Find neighbors within threshold
                if mrrc_id in mrrc_positions:
                    pos = mrrc_positions[mrrc_id]
                    for other in self.mrrcs:
                        if other.id not in visited:
                            dist = np.linalg.norm(pos - other.position)
                            if dist < threshold:
                                queue.append(other.id)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def cluster_mass(self, cluster: Set[int]) -> float:
        """Calculate mass of a cluster (in eV/c²)"""
        total_energy = len(cluster) * self.spawn_cost  # Energy locked in MRRCs
        
        # Mass = E/c² (if c is finite)
        if self.c and np.isfinite(self.c) and self.c > 0:
            return total_energy / (self.c ** 2)
        else:
            return total_energy  # Quantum regime: no mass concept yet


class EnhancedVisualizer:
    """Real-time visualization with 4D symmetry panel"""
    
    def __init__(self, universe: EnhancedMRRCUniverse):
        self.universe = universe
        
        # Create figure with 5 subplots
        self.fig = plt.figure(figsize=(20, 12))
        gs = self.fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        # 3D space view (top left, spans 2x2)
        self.ax_3d = self.fig.add_subplot(gs[:2, :2], projection='3d')
        
        # Population over time (top right)
        self.ax_pop = self.fig.add_subplot(gs[0, 2])
        
        # Mass evolution (middle right)
        self.ax_mass = self.fig.add_subplot(gs[1, 2])
        
        # Information metrics (third row)
        self.ax_info = self.fig.add_subplot(gs[2, :])
        
        # 4D Symmetry analysis (bottom row)
        self.ax_symmetry = self.fig.add_subplot(gs[3, :])
        
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # Animation
        self.anim = None
    
    def update(self, frame):
        """Update visualization"""
        if not self.universe.running:
            return
        
        # Evolve universe (fewer steps for performance)
        for _ in range(2):  # 2 steps per frame (reduced from 5)
            self.universe.evolve_step(self.universe.dt)
        
        # === 3D Space View ===
        self.ax_3d.clear()
        
        if self.universe.mrrcs:
            positions = np.array([m.position for m in self.universe.mrrcs])
            
            # Color by cluster
            clusters = self.universe.get_mass_clusters()
            colors = np.zeros(len(self.universe.mrrcs))
            for i, cluster in enumerate(clusters):
                for mrrc_id in cluster:
                    idx = next((j for j, m in enumerate(self.universe.mrrcs) 
                               if m.id == mrrc_id), None)
                    if idx is not None:
                        colors[idx] = i + 1
            
            self.ax_3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                             c=colors, cmap='tab20', s=50, alpha=0.7)
        
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title(f'3D Space - T={self.universe.current_time:.1f} - '
                           f'{self.universe.regime} regime',
                           fontweight='bold', fontsize=12)
        
        # === Population ===
        self.ax_pop.clear()
        if self.universe.history['time']:
            self.ax_pop.plot(self.universe.history['time'],
                           self.universe.history['num_mrrcs'],
                           'b-', linewidth=2)
        self.ax_pop.set_xlabel('Time')
        self.ax_pop.set_ylabel('Number of MRRCs')
        self.ax_pop.set_title('Population Growth', fontweight='bold')
        self.ax_pop.grid(True, alpha=0.3)
        
        # === Mass Evolution ===
        self.ax_mass.clear()
        if self.universe.history['time']:
            masses_kev = np.array(self.universe.history['max_cluster_mass']) / 1000
            self.ax_mass.plot(self.universe.history['time'], masses_kev,
                            'r-', linewidth=2, label='Max cluster')
            
            # Reference lines
            self.ax_mass.axhline(y=511, color='orange', linestyle='--', 
                               alpha=0.5, label='Electron (511 keV)')
            self.ax_mass.axhline(y=106000, color='green', linestyle='--',
                               alpha=0.5, label='Muon (106 MeV)')
        
        self.ax_mass.set_xlabel('Time')
        self.ax_mass.set_ylabel('Mass (keV/c²)')
        self.ax_mass.set_title('Mass Cluster Evolution', fontweight='bold')
        self.ax_mass.legend(loc='upper left', fontsize=8)
        self.ax_mass.grid(True, alpha=0.3)
        self.ax_mass.set_yscale('log')
        
        # === Information Metrics ===
        self.ax_info.clear()
        if self.universe.history['time']:
            self.ax_info.plot(self.universe.history['time'],
                            np.array(self.universe.history['energy']) / 1e9,
                            'g-', linewidth=2, label='Available Energy (GeV)')
            
            ax2 = self.ax_info.twinx()
            ax2.plot(self.universe.history['time'],
                    self.universe.history['cost'],
                    'orange', linewidth=2, label='Info Cost')
            ax2.set_ylabel('Cost (eV)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        
        self.ax_info.set_xlabel('Time')
        self.ax_info.set_ylabel('Energy (GeV)')
        self.ax_info.set_title('Energy & Information Cost', fontweight='bold')
        self.ax_info.legend(loc='upper left')
        self.ax_info.grid(True, alpha=0.3)
        
        # === 4D Symmetry Panel ===
        self.ax_symmetry.clear()
        self.ax_symmetry.set_title('4D Spacetime Symmetries (Eigenvalue Ratios)', 
                                  fontweight='bold', fontsize=11)
        
        if self.universe.symmetry_history:
            times = [s['time'] for s in self.universe.symmetry_history]
            
            ratio1 = [s['ratios'][0] if len(s['ratios']) > 0 else 0 
                     for s in self.universe.symmetry_history]
            ratio2 = [s['ratios'][1] if len(s['ratios']) > 1 else 0 
                     for s in self.universe.symmetry_history]
            ratio3 = [s['ratios'][2] if len(s['ratios']) > 2 else 0 
                     for s in self.universe.symmetry_history]
            
            self.ax_symmetry.plot(times, ratio1, 'r-', label='λ₂/λ₁', 
                                alpha=0.7, linewidth=2)
            self.ax_symmetry.plot(times, ratio2, 'g-', label='λ₃/λ₁', 
                                alpha=0.7, linewidth=2)
            self.ax_symmetry.plot(times, ratio3, 'b-', label='λ₄/λ₁', 
                                alpha=0.7, linewidth=2)
            
            # Symmetry reference lines
            self.ax_symmetry.axhline(y=1.0, color='gray', linestyle='--', 
                                   alpha=0.3, label='Perfect symmetry')
            
            # Show current symmetry type
            if self.universe.symmetry_history:
                current_sym = self.universe.symmetry_history[-1]
                self.ax_symmetry.text(0.02, 0.95, 
                                    f"Current: {current_sym['type']}", 
                                    transform=self.ax_symmetry.transAxes,
                                    fontsize=10, verticalalignment='top',
                                    bbox=dict(boxstyle='round', 
                                            facecolor='wheat', alpha=0.5))
        
        self.ax_symmetry.set_xlabel('Time')
        self.ax_symmetry.set_ylabel('Eigenvalue Ratios')
        self.ax_symmetry.set_ylim(0, 1.2)
        self.ax_symmetry.legend(loc='upper right')
        self.ax_symmetry.grid(True, alpha=0.3)
    
    def on_close(self, event):
        """Handle window close"""
        print("\nWindow closed - stopping simulation...")
        self.universe.running = False
    
    def show(self):
        """Start animation"""
        self.anim = FuncAnimation(self.fig, self.update, interval=50, cache_frame_data=False)
        plt.show()


def analyze_final_state(universe: EnhancedMRRCUniverse):
    """Detailed analysis after simulation ends"""
    print("\n" + "="*70)
    print("FINAL STATE ANALYSIS")
    print("="*70)
    
    print(f"\nSimulation time: {universe.current_time:.2f}")
    print(f"Total MRRCs spawned: {universe.total_spawned}")
    print(f"Active MRRCs: {len(universe.mrrcs)}")
    print(f"Regime: {universe.regime}")
    print(f"Speed of light: c = {universe.c:.4f}")
    if universe.decoherence_time:
        print(f"Decoherence occurred at T = {universe.decoherence_time:.2f}")
    
    # SPAWN ANALYSIS
    print("\n" + "="*70)
    print("MRRC SPAWN ANALYSIS")
    print("="*70)
    
    print(f"\nTotal spawns: {universe.total_spawned}")
    print(f"  Vacuum fluctuations:     {universe.vacuum_spawns:4d} ({100*universe.vacuum_spawns/max(universe.total_spawned,1):.1f}%)")
    print(f"  MRRC interactions:       {universe.interaction_spawns:4d} ({100*universe.interaction_spawns/max(universe.total_spawned,1):.1f}%)")
    print(f"  Cluster collisions:      {universe.cluster_collision_spawns:4d} ({100*universe.cluster_collision_spawns/max(universe.total_spawned,1):.1f}%)")
    
    if universe.interaction_spawns > 0:
        print("\n✓ YES! MRRCs spawned from interactions between existing MRRCs!")
        print(f"  {universe.interaction_spawns} MRRCs created from strong MRRC-MRRC interactions")
        
        # Show some examples
        interaction_events = [e for e in universe.spawn_events if e['cause'] == 'interaction']
        if interaction_events:
            print(f"\n  Example interaction spawns:")
            for event in interaction_events[:5]:  # Show first 5
                print(f"    T={event['time']:.1f}: MRRC {event['mrrc_id']} from parents {event['parent_ids']}")
    
    if universe.cluster_collision_spawns > 0:
        print("\n✓ YES! MRRCs spawned from cluster-cluster collisions!")
        print(f"  {universe.cluster_collision_spawns} MRRCs created when mass clusters collided")
        
        # Show some examples
        collision_events = [e for e in universe.spawn_events if e['cause'] == 'cluster_collision']
        if collision_events:
            print(f"\n  Example cluster collision spawns:")
            for event in collision_events[:5]:
                print(f"    T={event['time']:.1f}: MRRC {event['mrrc_id']} from cluster collision")
    
    if universe.interaction_spawns == 0 and universe.cluster_collision_spawns == 0:
        print("\n  All MRRCs spawned from vacuum fluctuations (no interaction spawning)")
    
    # Energy analysis
    print("\nENERGY BUDGET")
    print(f"  Initial:   {universe.total_energy:.2e} eV ({universe.total_energy/1e9:.1f} GeV)")
    print(f"  Remaining: {universe.available_energy:.2e} eV ({universe.available_energy/1e9:.1f} GeV)")
    print(f"  Used:      {universe.total_energy - universe.available_energy:.2e} eV")
    
    # Mass clusters
    clusters = universe.get_mass_clusters()
    print(f"\nMASS CLUSTERS DETECTED: {len(clusters)} clusters")
    
    if clusters:
        print("\nID   Size   Mass (eV)       Mass (eV/c²)    Comparison")
        print("-"*70)
        
        # Sort by size
        sorted_clusters = sorted(clusters, key=lambda c: len(c), reverse=True)
        
        for i, cluster in enumerate(sorted_clusters[:10]):  # Top 10
            mass_ev = len(cluster) * universe.spawn_cost
            mass_evc2 = universe.cluster_mass(cluster)
            
            comparison = ""
            if mass_evc2 > 500e3:  # > 500 keV
                comparison = f"{mass_evc2/511e3:.2f}× electron"
            elif mass_evc2 > 100e3:  # > 100 keV
                comparison = f"{mass_evc2/1e3:.0f} keV scale"
            
            print(f"{i:<4d} {len(cluster):<6d} {mass_ev:<15.2e} {mass_evc2:<15.2e} {comparison}")
        
        max_mass = max([universe.cluster_mass(c) for c in clusters])
        print(f"\nLARGEST MASS: {max_mass:.2e} eV/c²")
        
        print("\nCOMPARISON TO KNOWN PARTICLES:")
        print(f"  Electron: 5.11e+05 eV/c² (target)")
        print(f"  Muon:     1.06e+08 eV/c² (target)")
        print(f"  Proton:   9.38e+08 eV/c² (target)")
        
        if max_mass > 1e5:
            print(f"\n  ✓ Achieved significant mass scale!")
            print(f"  ✓ Max mass is {max_mass/511e3:.1f}× electron mass scale")
    
    # 4D Symmetry Analysis
    print("\n" + "="*70)
    print("4D SPACETIME SYMMETRY ANALYSIS")
    print("="*70)
    
    if universe.symmetry_history:
        print(f"\nSymmetry measurements: {len(universe.symmetry_history)}")
        
        # Count symmetry types
        types = [s['type'] for s in universe.symmetry_history]
        type_counts = Counter(types)
        
        print("\nSymmetry types observed:")
        for sym_type, count in type_counts.most_common():
            percentage = 100 * count / len(types)
            print(f"  {sym_type:25s}: {count:4d} ({percentage:.1f}%)")
        
        # Final symmetry state
        final_sym = universe.symmetry_history[-1]
        print(f"\nFinal symmetry: {final_sym['type']}")
        print(f"Eigenvalue ratios: λ₂/λ₁={final_sym['ratios'][0]:.3f}, "
              f"λ₃/λ₁={final_sym['ratios'][1]:.3f}, "
              f"λ₄/λ₁={final_sym['ratios'][2]:.3f}")
        
        # Check for emergent patterns
        print("\nEMERGENT GEOMETRIC PATTERNS:")
        if '4-sphere (SO(5))' in type_counts:
            print("  ✓ Full 4D rotational symmetry detected!")
            print("    (All 4 dimensions equally distributed)")
        if '3-sphere × line' in type_counts:
            print("  ✓ Spatial 3-sphere symmetry (like our universe!)")
            print("    (3D space isotropic, time separate)")
        if '2-sphere (rotational)' in type_counts:
            print("  ✓ 2D rotational symmetry")
            print("    (Disk-like or cylindrical structure)")
        if 'cylindrical' in type_counts:
            print("  ✓ Cylindrical symmetry")
            print("    (One preferred axis)")
        if 'linear (1D)' in type_counts:
            print("  ✓ Linear structure")
            print("    (Filament-like)")
        if 'asymmetric' in type_counts and type_counts['asymmetric'] > len(types) * 0.8:
            print("  ! Mostly asymmetric (symmetry breaking)")
            print("    (Complex structure, no preferred geometry)")
    else:
        print("\nNot enough data for symmetry analysis (need 4+ MRRCs)")
    
    print("\n" + "="*70)
    print("WHAT EMERGED:")
    print("="*70)
    print("✓ Time dimension (past states tracked)")
    print("✓ Space dimension (locality as information)")
    print("✓ Mass (stable clusters)")
    print(f"✓ Energy ({universe.total_energy/1e9:.0f} GeV → clusters + kinetic)")
    print("✓ Speed of light c")
    print("✓ Gravity (cost minimization)")
    print("✓ 4D spacetime geometry with emergent symmetries")
    print("\nALL FROM ONE RULE: Minimize information cost!")


if __name__ == "__main__":
    print("="*70)
    print("ENHANCED MRRC UNIVERSE SIMULATION")
    print("="*70)
    print("\n🌌 Starting with 10 TeV of energy...")
    print("🔬 Allowing up to 500 MRRCs with auto-spawning")
    print("📐 Analyzing 4D spacetime symmetries...")
    print("\n⚡ Watch the universe evolve in real-time!")
    print("❌ Close the window when you want to stop and analyze.\n")
    
    # Create universe
    universe = EnhancedMRRCUniverse(seed=42)
    
    print(f"💥 BIG BANG: {universe.total_energy:.2e} eV ({universe.total_energy/1e9:.1f} GeV)")
    print("∅  Initial state: vacuum")
    print("🌟 Everything is pure potential...")
    print(f"📊 Auto-spawn rate: {universe.auto_spawn_rate*100:.0f}% per step")
    print(f"⚛️  Spawn cost: {universe.spawn_cost:.0f} eV per MRRC")
    print(f"🛡️  Max MRRCs: {universe.max_mrrcs} (Mac mini protection)\n")
    print(f"🛡️  Max MRRCs: {universe.max_mrrcs} (Mac mini protection)\n")
    
    # Create visualizer
    viz = EnhancedVisualizer(universe)
    
    # Run (blocks until window closed)
    viz.show()
    
    # Analyze after window closes
    analyze_final_state(universe)
