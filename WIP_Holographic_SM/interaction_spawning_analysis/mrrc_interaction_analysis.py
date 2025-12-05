#!/usr/bin/env python3
"""
MRRC Interaction Spawning Analysis - No GUI version
Fast analysis of interaction-driven particle creation
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from collections import Counter

@dataclass
class MRRC:
    """Minimal Relational Reference Cell"""
    id: int
    position: np.ndarray
    state: str
    energy: float
    references: Set[int]
    birth_time: float
    
    def __post_init__(self):
        if not isinstance(self.references, set):
            self.references = set()

class MRRCUniverse:
    """Optimized universe for interaction analysis"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Energy budget
        self.total_energy = 1e13  # 10 TeV
        self.available_energy = self.total_energy
        
        # Constants
        self.spawn_cost = 500  # eV per MRRC
        self.interaction_cost_scale = 0.1
        self.cost_reduction_rate = 0.01
        self.auto_spawn_rate = 0.25  # Slightly reduced for stability
        self.max_mrrcs = 1200  # Balanced for Mac mini stability
        
        # MRRCs
        self.mrrcs: List[MRRC] = []
        self.next_id = 0
        
        # Time
        self.current_time = 0.0
        self.dt = 0.1
        
        # Phase tracking
        self.regime = "QUANTUM"
        self.decoherence_time = None
        self.information_capacity = 100
        self.c = float('inf')
        
        # Spawn tracking
        self.total_spawned = 0
        self.spawn_events = []
        self.vacuum_spawns = 0
        self.interaction_spawns = 0
        self.cluster_collision_spawns = 0
        
        # 4D symmetry tracking
        self.symmetry_history = []
    
    def spawn_mrrc(self, cause: str = "vacuum", position: Optional[np.ndarray] = None, 
                   parent_ids: Optional[List[int]] = None) -> Optional[MRRC]:
        """Create new MRRC"""
        if self.available_energy < self.spawn_cost:
            return None
        
        if position is not None:
            spawn_position = position
        elif self.mrrcs and np.random.random() < 0.7:
            target = self.mrrcs[np.random.randint(len(self.mrrcs))]
            spawn_position = target.position + np.random.randn(3) * 0.5
        else:
            spawn_position = np.random.randn(3) * 2.0
        
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
        
        # Vacuum spawning
        if self.available_energy >= self.spawn_cost and len(self.mrrcs) < self.max_mrrcs:
            if np.random.random() < self.auto_spawn_rate:
                self.spawn_mrrc(cause="vacuum")
        
        if len(self.mrrcs) < 2:
            return
        
        # Form references
        interactions_this_step = []
        for mrrc in self.mrrcs:
            for other in self.mrrcs:
                if other.id == mrrc.id:
                    continue
                
                dist = np.linalg.norm(mrrc.position - other.position)
                if dist < 3.0 and np.random.random() < 0.1:
                    mrrc.references.add(other.id)
                    interactions_this_step.append((mrrc.id, other.id, dist))
        
        # INTERACTION SPAWNING
        if self.available_energy >= self.spawn_cost and len(self.mrrcs) < self.max_mrrcs:
            for mrrc_id1, mrrc_id2, dist in interactions_this_step:
                mrrc1 = next((m for m in self.mrrcs if m.id == mrrc_id1), None)
                mrrc2 = next((m for m in self.mrrcs if m.id == mrrc_id2), None)
                
                if mrrc1 and mrrc2:
                    if dist < 0.5 and len(mrrc1.references) > 5 and len(mrrc2.references) > 5:
                        if np.random.random() < 0.12:  # 12% chance (reduced for stability)
                            midpoint = (mrrc1.position + mrrc2.position) / 2
                            offset = np.random.randn(3) * 0.2
                            new_mrrc = self.spawn_mrrc(
                                cause="interaction",
                                position=midpoint + offset,
                                parent_ids=[mrrc1.id, mrrc2.id]
                            )
                            if new_mrrc:
                                new_mrrc.references.add(mrrc1.id)
                                new_mrrc.references.add(mrrc2.id)
        
        # Apply gravity
        if len(self.mrrcs) >= 2:
            forces = [np.zeros(3) for _ in self.mrrcs]
            
            for i, mrrc in enumerate(self.mrrcs):
                for ref_id in mrrc.references:
                    ref_mrrc = next((m for m in self.mrrcs if m.id == ref_id), None)
                    if ref_mrrc:
                        direction = ref_mrrc.position - mrrc.position
                        dist = np.linalg.norm(direction)
                        if dist > 0.01:
                            force = self.cost_reduction_rate * direction / (dist + 0.1)
                            forces[i] += force
            
            for i, mrrc in enumerate(self.mrrcs):
                mrrc.position += forces[i] * dt
        
        # Decoherence check
        if self.regime == "QUANTUM":
            num_relationships = sum(len(m.references) for m in self.mrrcs)
            if num_relationships > self.information_capacity:
                self.regime = "CLASSICAL"
                self.decoherence_time = self.current_time
                avg_separation = np.mean([np.linalg.norm(m.position) 
                                         for m in self.mrrcs]) if self.mrrcs else 1.0
                self.c = max(avg_separation / dt, 1.0)
        
        # Analyze 4D symmetries every 10 steps
        if int(self.current_time / dt) % 10 == 0:
            self.analyze_4d_symmetries()
    
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
                symmetry['type'] = '4-sphere (SO(5))'
            elif np.allclose(ratios[:2], [1, 1], atol=0.1) and ratios[2] < 0.5:
                symmetry['type'] = '3-sphere × line'
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
        """Identify mass clusters using BFS (iterative)"""
        if len(self.mrrcs) < 2:
            return []
        
        visited = set()
        clusters = []
        
        # Build position lookup
        mrrc_positions = {m.id: m.position for m in self.mrrcs}
        
        for start_mrrc in self.mrrcs:
            if start_mrrc.id in visited:
                continue
            
            # BFS (no recursion)
            cluster = set()
            queue = [start_mrrc.id]
            
            while queue:
                mrrc_id = queue.pop(0)
                if mrrc_id in visited:
                    continue
                
                visited.add(mrrc_id)
                cluster.add(mrrc_id)
                
                # Find neighbors
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
        """Calculate mass of cluster in eV/c²"""
        total_energy = len(cluster) * self.spawn_cost
        
        if self.c and np.isfinite(self.c) and self.c > 0:
            return total_energy / (self.c ** 2)
        else:
            return total_energy
    
    def run(self, max_steps: int = 1000):
        """Run simulation for N steps"""
        print(f"Running up to {max_steps} steps...")
        print(f"Max MRRCs: {self.max_mrrcs}\n")
        
        for step in range(max_steps):
            self.evolve_step(self.dt)
            
            # Progress reporting
            if step % 100 == 0:
                clusters = self.get_mass_clusters()
                max_mass = max([self.cluster_mass(c) for c in clusters]) if clusters else 0
                print(f"  Step {step:4d}: {len(self.mrrcs):4d} MRRCs, "
                      f"V:{self.vacuum_spawns:3d}, "
                      f"I:{self.interaction_spawns:4d}, "
                      f"C:{len(clusters):2d}, "
                      f"MaxM: {max_mass:.2e} eV/c²")
            
            # Safety: stop if we hit max MRRCs
            if len(self.mrrcs) >= self.max_mrrcs:
                print(f"\n  ✓ Reached max MRRCs ({self.max_mrrcs}) at step {step}")
                break
            
            # Safety: stop if something went wrong
            if len(self.mrrcs) > self.max_mrrcs + 100:
                print(f"\n  ⚠ EMERGENCY STOP: Too many MRRCs ({len(self.mrrcs)})")
                break
        
        print("\nSimulation complete!\n")


def analyze(universe: MRRCUniverse):
    """Analyze final state"""
    print("="*70)
    print("INTERACTION SPAWNING ANALYSIS")
    print("="*70)
    
    print(f"\nSimulation time: {universe.current_time:.2f}")
    print(f"Final MRRC count: {len(universe.mrrcs)}")
    print(f"Regime: {universe.regime}")
    if universe.decoherence_time:
        print(f"Decoherence at T = {universe.decoherence_time:.2f}")
        print(f"Speed of light: c = {universe.c:.4f}")
    
    print("\n" + "="*70)
    print("SPAWN BREAKDOWN")
    print("="*70)
    
    total = universe.total_spawned
    print(f"\nTotal MRRCs spawned: {total}")
    print(f"  Vacuum fluctuations:     {universe.vacuum_spawns:5d} ({100*universe.vacuum_spawns/max(total,1):5.1f}%)")
    print(f"  MRRC interactions:       {universe.interaction_spawns:5d} ({100*universe.interaction_spawns/max(total,1):5.1f}%)")
    print(f"  Cluster collisions:      {universe.cluster_collision_spawns:5d} ({100*universe.cluster_collision_spawns/max(total,1):5.1f}%)")
    
    if universe.interaction_spawns > 0:
        print("\n" + "="*70)
        print("✓ YES! INTERACTION SPAWNING CONFIRMED")
        print("="*70)
        print(f"\n{universe.interaction_spawns} MRRCs created from MRRC-MRRC interactions!")
        print(f"That's {100*universe.interaction_spawns/max(total,1):.1f}% of all MRRCs!")
        
        # Show examples
        interaction_events = [e for e in universe.spawn_events if e['cause'] == 'interaction']
        if interaction_events:
            print(f"\nExample interaction spawns (first 10):")
            for event in interaction_events[:10]:
                print(f"  T={event['time']:6.1f}: MRRC {event['mrrc_id']:4d} from parents {event['parent_ids']}")
            
            # Analyze spawn rate over time
            time_bins = [0, 10, 20, 30, 40, 50, 100, 200, float('inf')]
            bin_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-100', '100-200', '200+']
            
            print(f"\nInteraction spawns over time:")
            for i in range(len(time_bins)-1):
                count = sum(1 for e in interaction_events 
                           if time_bins[i] <= e['time'] < time_bins[i+1])
                if count > 0:
                    print(f"  T={bin_labels[i]:8s}: {count:4d} spawns")
    
    # Mass cluster analysis
    print("\n" + "="*70)
    print("MASS CLUSTER ANALYSIS")
    print("="*70)
    
    clusters = universe.get_mass_clusters()
    print(f"\nClusters detected: {len(clusters)}")
    
    if clusters:
        # Sort by size
        sorted_clusters = sorted(clusters, key=lambda c: len(c), reverse=True)
        
        print(f"\nTop 10 largest clusters:")
        print(f"{'ID':<4} {'Size':<6} {'Mass (eV)':<15} {'Mass (eV/c²)':<15} {'Comparison'}")
        print("-"*70)
        
        for i, cluster in enumerate(sorted_clusters[:10]):
            mass_ev = len(cluster) * universe.spawn_cost
            mass_evc2 = universe.cluster_mass(cluster)
            
            comparison = ""
            if mass_evc2 > 900e6:  # > 900 MeV
                comparison = f"{mass_evc2/938e6:.2f}× proton"
            elif mass_evc2 > 100e6:  # > 100 MeV
                comparison = f"{mass_evc2/106e6:.2f}× muon"
            elif mass_evc2 > 500e3:  # > 500 keV
                comparison = f"{mass_evc2/511e3:.2f}× electron"
            elif mass_evc2 > 100e3:
                comparison = f"{mass_evc2/1e3:.0f} keV scale"
            
            print(f"{i:<4d} {len(cluster):<6d} {mass_ev:<15.2e} {mass_evc2:<15.2e} {comparison}")
        
        max_mass = max([universe.cluster_mass(c) for c in clusters])
        print(f"\nLargest mass: {max_mass:.2e} eV/c²")
        
        # Reference particles
        print("\nReference particle masses:")
        print(f"  Electron: 5.11e+05 eV/c²")
        print(f"  Muon:     1.06e+08 eV/c²")
        print(f"  Proton:   9.38e+08 eV/c²")
        
        if max_mass > 1e6:
            print(f"\n✓ Achieved MeV scale! ({max_mass/1e6:.1f} MeV)")
        if max_mass > 500e3:
            print(f"✓ Approaching electron mass! ({max_mass/511e3:.2f}× electron)")
    
    # 4D Symmetry analysis
    print("\n" + "="*70)
    print("4D SPACETIME SYMMETRY ANALYSIS")
    print("="*70)
    
    if universe.symmetry_history:
        print(f"\nSymmetry measurements: {len(universe.symmetry_history)}")
        
        # Count types
        types = [s['type'] for s in universe.symmetry_history]
        type_counts = Counter(types)
        
        print("\nSymmetry types observed:")
        for sym_type, count in type_counts.most_common():
            percentage = 100 * count / len(types)
            print(f"  {sym_type:25s}: {count:4d} ({percentage:5.1f}%)")
        
        # Final state
        final_sym = universe.symmetry_history[-1]
        print(f"\nFinal symmetry: {final_sym['type']}")
        print(f"Eigenvalue ratios: λ₂/λ₁={final_sym['ratios'][0]:.3f}, "
              f"λ₃/λ₁={final_sym['ratios'][1]:.3f}, "
              f"λ₄/λ₁={final_sym['ratios'][2]:.3f}")
        
        print("\nEmergent geometric patterns:")
        if '4-sphere (SO(5))' in type_counts:
            print("  ✓ Full 4D rotational symmetry detected!")
        if '3-sphere × line' in type_counts:
            print("  ✓ Spatial 3-sphere symmetry (like our universe!)")
        if '2-sphere (rotational)' in type_counts:
            print("  ✓ 2D rotational symmetry")
        if 'asymmetric' in type_counts and type_counts['asymmetric'] > len(types) * 0.5:
            print("  ! Symmetry breaking dominant (complex structure)")
    
    # Energy analysis
    print("\n" + "="*70)
    print("ENERGY BUDGET")
    print("="*70)
    used = universe.total_energy - universe.available_energy
    print(f"  Initial:   {universe.total_energy:.2e} eV ({universe.total_energy/1e9:.0f} GeV)")
    print(f"  Used:      {used:.2e} eV ({100*used/universe.total_energy:.2f}%)")
    print(f"  Remaining: {universe.available_energy:.2e} eV")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    if universe.interaction_spawns > universe.vacuum_spawns:
        print("\n🎯 INTERACTIONS DOMINATE PARTICLE CREATION!")
        print(f"   {universe.interaction_spawns}× more interaction spawns than vacuum")
        print("\n   This mirrors real physics:")
        print("   - Particle accelerators: collisions → new particles")
        print("   - Force carriers: interactions → photons, gluons")
        print("   - Pair production: high energy → matter + antimatter")
        print("\n   In MRRC: Information cost minimization → new MRRCs")
    else:
        print("\n   Vacuum fluctuations dominate (universe too sparse)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("="*70)
    print("MRRC INTERACTION SPAWNING ANALYSIS")
    print("="*70)
    print("\n🌌 10 TeV universe")
    print("🔬 Tracking spawn causes")
    print("⚛️  No visualization (pure computation)\n")
    
    universe = MRRCUniverse(seed=42)
    
    print(f"💥 BIG BANG: {universe.total_energy:.2e} eV\n")
    
    # Run simulation
    universe.run(max_steps=3000)
    
    # Analyze results
    analyze(universe)
