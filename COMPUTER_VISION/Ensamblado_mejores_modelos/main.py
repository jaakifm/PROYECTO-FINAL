import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timm
import random
import time
import os

# 1. ALGORITMO GENÉTICO
class GeneticAlgorithm:
    def __init__(self, test_loader, vit_model, efficientnet_model, device, 
                 population_size=20, generations=10, mutation_rate=0.1, crossover_rate=0.8, 
                 num_classes=2, elitism=2):
        self.test_loader = test_loader
        self.vit_model = vit_model
        self.efficientnet_model = efficientnet_model
        self.device = device
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_classes = num_classes
        self.elitism = elitism
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
    def initialize_population(self):
        """Inicializa la población con cromosomas (pesos aleatorios)"""
        population = []
        for _ in range(self.population_size):
            # Generar un peso aleatorio para ViT entre 0 y 1
            vit_weight = random.uniform(0, 1)
            # El peso de EfficientNet complementa al de ViT para sumar 1
            efficientnet_weight = 1 - vit_weight
            # Cada cromosoma es un par (vit_weight, efficientnet_weight)
            population.append((vit_weight, efficientnet_weight))
        return population
    
    def evaluate_fitness(self, chromosome):
        """Evalúa la aptitud (fitness) de un cromosoma midiendo el F1 score del ensemble"""
        vit_weight, efficientnet_weight = chromosome
        
        # Crear el modelo de ensemble con estos pesos
        ensemble_model = EnsembleModel(
            self.vit_model, 
            self.efficientnet_model, 
            vit_weight=vit_weight, 
            efficientnet_weight=efficientnet_weight
        ).to(self.device)
        
        # Evaluar el modelo
        ensemble_model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Predicciones del ensemble
                outputs = ensemble_model(inputs)
                
                # Procesar predicciones según sea binario o multiclase
                if self.num_classes == 2:
                    if outputs.shape[1] > 1:  # Si la salida es [batch_size, 2]
                        preds = (outputs[:, 1] > 0.5).float()
                    else:  # Si la salida es [batch_size, 1]
                        preds = (outputs.view(-1) > 0.5).float()
                else:
                    preds = torch.argmax(outputs, dim=1)
                
                # Asegurar forma correcta
                preds = preds.view(-1)
                labels = labels.view(-1)
                
                # Guardar predicciones y etiquetas verdaderas
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
        
        # Calcular F1 score
        f1 = f1_score(all_true, all_preds, average='weighted')
        return f1
    
    def select_parents(self, population, fitness_scores):
        """Selecciona padres para reproducción usando selección por torneos"""
        def tournament_selection(tournament_size=3):
            # Seleccionar aleatoriamente individuos para el torneo
            candidates = random.sample(range(len(population)), tournament_size)
            # Encontrar el mejor candidato
            best_idx = candidates[0]
            for idx in candidates:
                if fitness_scores[idx] > fitness_scores[best_idx]:
                    best_idx = idx
            return population[best_idx]
        
        # Seleccionar dos padres
        parent1 = tournament_selection()
        parent2 = tournament_selection()
        return parent1, parent2
    
    def crossover(self, parent1, parent2):
        """Realiza el cruce entre dos padres"""
        if random.random() < self.crossover_rate:
            # Cruce aritmético: tomar una combinación ponderada de los padres
            alpha = random.random()
            vit_weight = alpha * parent1[0] + (1 - alpha) * parent2[0]
            efficientnet_weight = 1 - vit_weight  # Asegurar que suman 1
            child = (vit_weight, efficientnet_weight)
        else:
            # Sin cruce, devolver uno de los padres
            child = parent1 if random.random() < 0.5 else parent2
        return child
    
    def mutate(self, chromosome):
        """Aplica mutación a un cromosoma"""
        vit_weight, efficientnet_weight = chromosome
        if random.random() < self.mutation_rate:
            # Añadir una perturbación gaussiana al peso de ViT
            mutation = random.gauss(0, 0.1)  # Media 0, desviación estándar 0.1
            vit_weight = max(0, min(1, vit_weight + mutation))  # Mantener dentro de [0, 1]
            efficientnet_weight = 1 - vit_weight  # Asegurar que suman 1
        return (vit_weight, efficientnet_weight)
    
    def evolve(self):
        """Evoluciona la población a través de múltiples generaciones"""
        start_time = time.time()
        
        # Inicializar población
        population = self.initialize_population()
        
        for generation in range(self.generations):
            gen_start_time = time.time()
            
            # Evaluar fitness de cada cromosoma en la población
            fitness_scores = [self.evaluate_fitness(chromosome) for chromosome in population]
            
            # Guardar el mejor de esta generación
            current_best_idx = fitness_scores.index(max(fitness_scores))
            current_best_fitness = fitness_scores[current_best_idx]
            current_best_solution = population[current_best_idx]
            
            # Actualizar el mejor global si es necesario
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = current_best_solution
            
            # Guardar historial de fitness
            self.fitness_history.append(current_best_fitness)
            
            # Imprimir progreso
            gen_time = time.time() - gen_start_time
            print(f"Generación {generation+1}/{self.generations}: Mejor F1 = {current_best_fitness:.4f}, "
                  f"Mejor global = {self.best_fitness:.4f}, Tiempo: {gen_time:.2f}s")
            
            # Crear nueva población
            new_population = []
            
            # Elitismo: conservar los mejores individuos sin cambios
            if self.elitism > 0:
                # Ordenar población por fitness en orden descendente
                sorted_indices = np.argsort(fitness_scores)[::-1]
                for i in range(self.elitism):
                    if i < len(sorted_indices):
                        elite_idx = sorted_indices[i]
                        new_population.append(population[elite_idx])
            
            # Generar el resto de la población mediante selección, cruce y mutación
            while len(new_population) < self.population_size:
                # Seleccionar padres
                parent1, parent2 = self.select_parents(population, fitness_scores)
                
                # Cruce
                child = self.crossover(parent1, parent2)
                
                # Mutación
                child = self.mutate(child)
                
                # Añadir a la nueva población
                new_population.append(child)
            
            # Reemplazar la población antigua con la nueva
            population = new_population
        
        total_time = time.time() - start_time
        print(f"\nAlgoritmo Genético completado en {total_time:.2f} segundos")
        print(f"Mejor solución encontrada: ViT Weight = {self.best_solution[0]:.4f}, "
              f"EfficientNet Weight = {self.best_solution[1]:.4f}")
        print(f"Mejor F1 Score: {self.best_fitness:.4f}")
        
        return self.best_solution, self.best_fitness, self.fitness_history

# 2. PARTICLE SWARM OPTIMIZATION (PSO)
class ParticleSwarmOptimization:
    def __init__(self, test_loader, vit_model, efficientnet_model, device, 
                 num_particles=20, iterations=10, c1=1.5, c2=1.5, w=0.7, 
                 num_classes=2):
        self.test_loader = test_loader
        self.vit_model = vit_model
        self.efficientnet_model = efficientnet_model
        self.device = device
        self.num_particles = num_particles
        self.iterations = iterations
        self.c1 = c1  # Coeficiente cognitivo
        self.c2 = c2  # Coeficiente social
        self.w = w    # Inercia
        self.num_classes = num_classes
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.fitness_history = []
    
    def initialize_particles(self):
        """Inicializa las partículas con posiciones y velocidades aleatorias"""
        particles = []
        for _ in range(self.num_particles):
            # Posición inicial: un peso aleatorio para ViT entre 0 y 1
            position = random.uniform(0, 1)
            # Velocidad inicial aleatoria
            velocity = random.uniform(-0.1, 0.1)
            # Cada partícula es (posición, velocidad, mejor_posición_personal, mejor_fitness_personal)
            particles.append({
                'position': position,  # Representa el peso de ViT (el de EfficientNet es 1-position)
                'velocity': velocity,
                'best_position': position,
                'best_fitness': -float('inf')
            })
        return particles
    
    def evaluate_fitness(self, position):
        """Evalúa la aptitud (fitness) de una posición midiendo el F1 score del ensemble"""
        vit_weight = position
        efficientnet_weight = 1 - vit_weight
        
        # Crear el modelo de ensemble con estos pesos
        ensemble_model = EnsembleModel(
            self.vit_model, 
            self.efficientnet_model, 
            vit_weight=vit_weight, 
            efficientnet_weight=efficientnet_weight
        ).to(self.device)
        
        # Evaluar el modelo
        ensemble_model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Predicciones del ensemble
                outputs = ensemble_model(inputs)
                
                # Procesar predicciones según sea binario o multiclase
                if self.num_classes == 2:
                    if outputs.shape[1] > 1:  # Si la salida es [batch_size, 2]
                        preds = (outputs[:, 1] > 0.5).float()
                    else:  # Si la salida es [batch_size, 1]
                        preds = (outputs.view(-1) > 0.5).float()
                else:
                    preds = torch.argmax(outputs, dim=1)
                
                # Asegurar forma correcta
                preds = preds.view(-1)
                labels = labels.view(-1)
                
                # Guardar predicciones y etiquetas verdaderas
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
        
        # Calcular F1 score
        f1 = f1_score(all_true, all_preds, average='weighted')
        return f1
    
    def update_particle(self, particle, global_best_position):
        """Actualiza la velocidad y posición de una partícula"""
        # Componentes aleatorios
        r1 = random.random()
        r2 = random.random()
        
        # Actualizar velocidad (fórmula PSO estándar)
        cognitive_component = self.c1 * r1 * (particle['best_position'] - particle['position'])
        social_component = self.c2 * r2 * (global_best_position - particle['position'])
        
        particle['velocity'] = self.w * particle['velocity'] + cognitive_component + social_component
        
        # Limitar velocidad para evitar explosiones
        particle['velocity'] = max(-0.2, min(0.2, particle['velocity']))
        
        # Actualizar posición
        particle['position'] += particle['velocity']
        
        # Mantener la posición dentro del rango [0, 1]
        particle['position'] = max(0, min(1, particle['position']))
        
        return particle
    
    def optimize(self):
        """Ejecuta el algoritmo PSO"""
        start_time = time.time()
        
        # Inicializar partículas
        particles = self.initialize_particles()
        global_best_position = 0.5  # Posición inicial arbitraria
        
        for iteration in range(self.iterations):
            iter_start_time = time.time()
            
            for i, particle in enumerate(particles):
                # Evaluar fitness actual
                fitness = self.evaluate_fitness(particle['position'])
                
                # Actualizar mejor posición personal si es necesario
                if fitness > particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = particle['position']
                
                # Actualizar mejor posición global si es necesario
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = particle['position']
                    global_best_position = particle['position']
            
            # Guardar mejor fitness de esta iteración
            self.fitness_history.append(self.best_fitness)
            
            # Actualizar todas las partículas
            for i in range(len(particles)):
                particles[i] = self.update_particle(particles[i], global_best_position)
            
            # Imprimir progreso
            iter_time = time.time() - iter_start_time
            print(f"Iteración {iteration+1}/{self.iterations}: Mejor F1 = {self.best_fitness:.4f}, "
                  f"Tiempo: {iter_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"\nPSO completado en {total_time:.2f} segundos")
        print(f"Mejor solución encontrada: ViT Weight = {self.best_solution:.4f}, "
              f"EfficientNet Weight = {1-self.best_solution:.4f}")
        print(f"Mejor F1 Score: {self.best_fitness:.4f}")
        
        return self.best_solution, 1-self.best_solution, self.best_fitness, self.fitness_history

# 3. SIMULATED ANNEALING
class SimulatedAnnealing:
    def __init__(self, test_loader, vit_model, efficientnet_model, device, 
                 initial_temp=100, cooling_rate=0.95, iterations=50, num_classes=2):
        self.test_loader = test_loader
        self.vit_model = vit_model
        self.efficientnet_model = efficientnet_model
        self.device = device
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.num_classes = num_classes
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.fitness_history = []
    
    def evaluate_fitness(self, vit_weight):
        """Evalúa la aptitud (fitness) de un punto midiendo el F1 score del ensemble"""
        efficientnet_weight = 1 - vit_weight
        
        # Crear el modelo de ensemble con estos pesos
        ensemble_model = EnsembleModel(
            self.vit_model, 
            self.efficientnet_model, 
            vit_weight=vit_weight, 
            efficientnet_weight=efficientnet_weight
        ).to(self.device)
        
        # Evaluar el modelo
        ensemble_model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Predicciones del ensemble
                outputs = ensemble_model(inputs)
                
                # Procesar predicciones según sea binario o multiclase
                if self.num_classes == 2:
                    if outputs.shape[1] > 1:  # Si la salida es [batch_size, 2]
                        preds = (outputs[:, 1] > 0.5).float()
                    else:  # Si la salida es [batch_size, 1]
                        preds = (outputs.view(-1) > 0.5).float()
                else:
                    preds = torch.argmax(outputs, dim=1)
                
                # Asegurar forma correcta
                preds = preds.view(-1)
                labels = labels.view(-1)
                
                # Guardar predicciones y etiquetas verdaderas
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
        
        # Calcular F1 score
        f1 = f1_score(all_true, all_preds, average='weighted')
        return f1
    
    def get_neighbor(self, current_state, temp):
        """Genera un estado vecino con una perturbación proporcional a la temperatura"""
        # La perturbación es más grande cuando la temperatura es alta
        perturbation = random.gauss(0, 0.1 * temp / self.initial_temp)
        # Generar un nuevo estado añadiendo la perturbación
        new_state = current_state + perturbation
        # Asegurar que el nuevo estado está en el rango [0, 1]
        new_state = max(0, min(1, new_state))
        return new_state
    
    def acceptance_probability(self, old_fitness, new_fitness, temperature):
        """Calcula la probabilidad de aceptar un nuevo estado"""
        # Si el nuevo estado es mejor, lo aceptamos siempre
        if new_fitness > old_fitness:
            return 1.0
        
        # Si es peor, lo aceptamos con una probabilidad que depende de cuánto peor es
        # y de la temperatura actual
        return np.exp((new_fitness - old_fitness) / temperature)
    
    def optimize(self):
        """Ejecuta el algoritmo de recocido simulado"""
        start_time = time.time()
        
        # Estado inicial: un peso aleatorio para ViT entre 0 y 1
        current_state = random.uniform(0, 1)
        current_fitness = self.evaluate_fitness(current_state)
        
        # Inicializar el mejor estado
        self.best_solution = current_state
        self.best_fitness = current_fitness
        
        # Temperatura inicial
        temperature = self.initial_temp
        
        for iteration in range(self.iterations):
            iter_start_time = time.time()
            
            # Generar un nuevo estado vecino
            neighbor_state = self.get_neighbor(current_state, temperature)
            neighbor_fitness = self.evaluate_fitness(neighbor_state)
            
            # Decidir si aceptamos el nuevo estado
            ap = self.acceptance_probability(current_fitness, neighbor_fitness, temperature)
            if random.random() < ap:
                current_state = neighbor_state
                current_fitness = neighbor_fitness
                
                # Actualizar el mejor estado si es necesario
                if current_fitness > self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_solution = current_state
            
            # Guardar mejor fitness de esta iteración
            self.fitness_history.append(self.best_fitness)
            
            # Enfriar la temperatura
            temperature *= self.cooling_rate
            
            # Imprimir progreso
            iter_time = time.time() - iter_start_time
            print(f"Iteración {iteration+1}/{self.iterations}: "
                  f"Temp = {temperature:.2f}, "
                  f"Estado actual = {current_state:.4f}, "
                  f"Fitness actual = {current_fitness:.4f}, "
                  f"Mejor = {self.best_fitness:.4f}, "
                  f"Tiempo: {iter_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"\nRecocido Simulado completado en {total_time:.2f} segundos")
        print(f"Mejor solución encontrada: ViT Weight = {self.best_solution:.4f}, "
              f"EfficientNet Weight = {1-self.best_solution:.4f}")
        print(f"Mejor F1 Score: {self.best_fitness:.4f}")
        
        return self.best_solution, 1-self.best_solution, self.best_fitness, self.fitness_history

# 4. GRID SEARCH
class GridSearch:
    def __init__(self, test_loader, vit_model, efficientnet_model, device, 
                 grid_resolution=20, num_classes=2):
        self.test_loader = test_loader
        self.vit_model = vit_model
        self.efficientnet_model = efficientnet_model
        self.device = device
        self.grid_resolution = grid_resolution
        self.num_classes = num_classes
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.all_results = []
    
    def evaluate_fitness(self, vit_weight):
        """Evalúa la aptitud (fitness) de un punto midiendo el F1 score del ensemble"""
        efficientnet_weight = 1 - vit_weight
        
        # Crear el modelo de ensemble con estos pesos
        ensemble_model = EnsembleModel(
            self.vit_model, 
            self.efficientnet_model, 
            vit_weight=vit_weight, 
            efficientnet_weight=efficientnet_weight
        ).to(self.device)
        
        # Evaluar el modelo
        ensemble_model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Predicciones del ensemble
                outputs = ensemble_model(inputs)
                
                # Procesar predicciones según sea binario o multiclase
                if self.num_classes == 2:
                    if outputs.shape[1] > 1:  # Si la salida es [batch_size, 2]
                        preds = (outputs[:, 1] > 0.5).float()
                    else:  # Si la salida es [batch_size, 1]
                        preds = (outputs.view(-1) > 0.5).float()
                else:
                    preds = torch.argmax(outputs, dim=1)
                
                # Asegurar forma correcta
                preds = preds.view(-1)
                labels = labels.view(-1)
                
                # Guardar predicciones y etiquetas verdaderas
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
        
        # Calcular F1 score
        f1 = f1_score(all_true, all_preds, average='weighted')
        return f1
    
    def search(self):
        """Ejecuta la búsqueda de cuadrícula"""
        start_time = time.time()
        
        # Generar la cuadrícula de pesos para ViT
        grid = np.linspace(0, 1, self.grid_resolution)
        
        for i, vit_weight in enumerate(grid):
            iter_start_time = time.time()
            
            # Evaluar fitness
            fitness = self.evaluate_fitness(vit_weight)
            
            # Guardar resultado
            self.all_results.append((vit_weight, 1-vit_weight, fitness))
            
            # Actualizar mejor solución si es necesario
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = (vit_weight, 1-vit_weight)
            
            # Imprimir progreso
            iter_time = time.time() - iter_start_time
            print(f"Grid point {i+1}/{self.grid_resolution}: "
                  f"ViT Weight = {vit_weight:.4f}, "
                  f"Fitness = {fitness:.4f}, "
                  f"Mejor = {self.best_fitness:.4f}, "
                  f"Tiempo: {iter_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"\nGrid Search completado en {total_time:.2f} segundos")
        print(f"Mejor solución encontrada: ViT Weight = {self.best_solution[0]:.4f}, "
              f"EfficientNet Weight = {self.best_solution[1]:.4f}")
        print(f"Mejor F1 Score: {self.best_fitness:.4f}")
        
        return self.best_solution, self.best_fitness, self.all_results

# FUNCIÓN PRINCIPAL QUE EJECUTA Y COMPARA TODOS LOS MÉTODOS METAHEURÍSTICOS
def optimize_ensemble_weights(test_loader, vit_model, efficientnet_model, device, num_classes=2):
    """Ejecuta y compara todos los métodos metaheurísticos"""
    results = {}
    
    # Configuraciones para cada método
    # (Reducidas para prueba de concepto - aumentar para mejores resultados)
    ga_config = {
        'population_size': 10, 
        'generations': 5, 
        'mutation_rate': 0.1, 
        'crossover_rate': 0.8,
        'elitism': 2
    }
    
    pso_config = {
        'num_particles': 10, 
        'iterations': 5, 
        'c1': 1.5, 
        'c2': 1.5, 
        'w': 0.7
    }
    
    sa_config = {
        'initial_temp': 100, 
        'cooling_rate': 0.9, 
        'iterations': 10
    }
    
    gs_config = {
        'grid_resolution': 11  # 0.0, 0.1, 0.2, ..., 1.0
    }
    
    # 1. Algoritmo Genético
    print("\n=== EJECUTANDO ALGORITMO GENÉTICO ===")
    ga = GeneticAlgorithm(
        test_loader, vit_model, efficientnet_model, device, 
        population_size=ga_config['population_size'],
        generations=ga_config['generations'],
        mutation_rate=ga_config['mutation_rate'],
        crossover_rate=ga_config['crossover_rate'],
        num_classes=num_classes,
        elitism=ga_config['elitism']
    )
    ga_solution, ga_fitness, ga_history = ga.evolve()
    results['GA'] = {
        'solution': ga_solution,
        'fitness': ga_fitness,
        'history': ga_history
    }
    
    # 2. Particle Swarm Optimization
    print("\n=== EJECUTANDO PARTICLE SWARM OPTIMIZATION ===")
    pso = ParticleSwarmOptimization(
        test_loader, vit_model, efficientnet_model, device,
        num_particles=pso_config['num_particles'],
        iterations=pso_config['iterations'],
        c1=pso_config['c1'],
        c2=pso_config['c2'],
        w=pso_config['w'],
        num_classes=num_classes
    )
    pso_vit_weight, pso_efficientnet_weight, pso_fitness, pso_history = pso.optimize()
    results['PSO'] = {
        'solution': (pso_vit_weight, pso_efficientnet_weight),
        'fitness': pso_fitness,
        'history': pso_history
    }
    
    # 3. Simulated Annealing
    print("\n=== EJECUTANDO SIMULATED ANNEALING ===")
    sa = SimulatedAnnealing(
        test_loader, vit_model, efficientnet_model, device,
        initial_temp=sa_config['initial_temp'],
        cooling_rate=sa_config['cooling_rate'],
        iterations=sa_config['iterations'],
        num_classes=num_classes
    )
    sa_vit_weight, sa_efficientnet_weight, sa_fitness, sa_history = sa.optimize()
    results['SA'] = {
        'solution': (sa_vit_weight, sa_efficientnet_weight),
        'fitness': sa_fitness,
        'history': sa_history
    }
    
    # 4. Grid Search
    print("\n=== EJECUTANDO GRID SEARCH ===")
    gs = GridSearch(
        test_loader, vit_model, efficientnet_model, device,
        grid_resolution=gs_config['grid_resolution'],
        num_classes=num_classes
    )
    gs_solution, gs_fitness, gs_all_results = gs.search()
    results['GS'] = {
        'solution': gs_solution,
        'fitness': gs_fitness,
        'all_results': gs_all_results
    }
    
    # Comparación de resultados
    print("\n=== COMPARACIÓN DE MÉTODOS METAHEURÍSTICOS ===")
    print(f"{'Método':<20} {'ViT Weight':<15} {'EfficientNet Weight':<20} {'F1 Score':<10}")
    print("-" * 65)
    
    for method, data in results.items():
        if method == 'GA':
            vit_w, eff_w = data['solution']
        else:
            vit_w, eff_w = data['solution']
        
        print(f"{method:<20} {vit_w:.4f}{' ':<10} {eff_w:.4f}{' ':<15} {data['fitness']:.4f}")
    
    # Encontrar el mejor método
    best_method = max(results.items(), key=lambda x: x[1]['fitness'])[0]
    best_solution = results[best_method]['solution']
    best_fitness = results[best_method]['fitness']
    
    print("\n=== MEJOR CONFIGURACIÓN ENCONTRADA ===")
    print(f"Método: {best_method}")
    print(f"ViT Weight: {best_solution[0]:.4f}")
    print(f"EfficientNet Weight: {best_solution[1]:.4f}")
    print(f"F1 Score: {best_fitness:.4f}")
    
    # Visualizar resultados
    plt.figure(figsize=(15, 10))
    
    # 1. Gráfico de convergencia para métodos iterativos
    plt.subplot(2, 2, 1)
    for method in ['GA', 'PSO', 'SA']:
        if method in results:
            history = results[method]['history']
            plt.plot(range(1, len(history) + 1), history, label=method)
    plt.xlabel('Iteración')
    plt.ylabel('F1 Score')
    plt.title('Convergencia de métodos iterativos')
    plt.legend()
    plt.grid(True)
    
    # 2. Resultados de Grid Search
    plt.subplot(2, 2, 2)
    if 'GS' in results:
        gs_results = results['GS']['all_results']
        vit_weights = [r[0] for r in gs_results]
        f1_scores = [r[2] for r in gs_results]
        plt.plot(vit_weights, f1_scores, 'o-')
        plt.xlabel('ViT Weight')
        plt.ylabel('F1 Score')
        plt.title('Resultados de Grid Search')
        plt.grid(True)
    
    # 3. Comparación de los mejores resultados
    plt.subplot(2, 2, 3)
    methods = list(results.keys())
    fitnesses = [results[m]['fitness'] for m in methods]
    plt.bar(methods, fitnesses)
    plt.xlabel('Método')
    plt.ylabel('F1 Score')
    plt.title('Comparación de F1 Score entre métodos')
    for i, v in enumerate(fitnesses):
        plt.text(i, v+0.005, f"{v:.4f}", ha='center')
    
    # 4. Distribución de pesos óptimos
    plt.subplot(2, 2, 4)
    vit_weights = [results[m]['solution'][0] for m in methods]
    plt.bar(methods, vit_weights)
    plt.xlabel('Método')
    plt.ylabel('ViT Weight')
    plt.title('Pesos óptimos para ViT por método')
    for i, v in enumerate(vit_weights):
        plt.text(i, v+0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('metaheuristic_comparison.png')
    plt.show()
    
    # Devolver el mejor modelo ensamblado con los pesos optimizados
    best_vit_weight, best_efficientnet_weight = best_solution
    best_ensemble = EnsembleModel(
        vit_model, 
        efficientnet_model, 
        vit_weight=best_vit_weight, 
        efficientnet_weight=best_efficientnet_weight
    ).to(device)
    
    return best_ensemble, best_method, best_solution, best_fitness

# CÓDIGO PRINCIPAL DE EJECUCIÓN
if __name__ == "__main__":
    # Definir el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar los modelos pre-entrenados definidos en tu código original
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as transforms
    
    # Rutas a los modelos pre-entrenados
    VIT_MODEL_PATH = r"C:\Users\jakif\CODE\PROYECTO-FINAL\COMPUTER_VISION\vision_transformers\best_vit_model.pth"
    EFFICIENT_MODEL_PATH = r"C:\Users\jakif\CODE\PROYECTO-FINAL\COMPUTER_VISION\melanoma_model_1_torch_EFFICIENTNETB0_harvard_attention.pth"
    
    # Ruta al conjunto de datos de test
    path_test = r"C:\Users\jakif\CODE\PROYECTO-FINAL\images\PREPROCESSED_DATA_copy\test"
    
    # Definir transformaciones para el conjunto de test
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Cargar conjunto de datos de test
    test_dataset = ImageFolder(root=path_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Cargar los modelos pre-entrenados
    print("Cargando modelos pre-entrenados...")
    num_classes = 2  # Clasificación binaria
    
    # Recreamos las funciones necesarias del código original

    def create_vit_classifier(num_classes=2, dropout_rate=0.2):
        # Use vit_large_patch16_224 to match the checkpoint dimensions
        model = timm.create_model('vit_large_patch16_224', pretrained=True, drop_rate=dropout_rate)
        
        # Modify the classification layer
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        return model
        
    class SEBlock(nn.Module):
        def __init__(self, channel, reduction=16):
            super(SEBlock, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze: Global Average Pooling
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),  # Reducción de dimensionalidad
                nn.ReLU(inplace=True),  # Activación
                nn.Linear(channel // reduction, channel, bias=False),  # Restauración de dimensionalidad
                nn.Sigmoid()  # Escala entre 0 y 1 para recalibración
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            # Squeeze
            y = self.avg_pool(x).view(b, c)  # [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
            # Excitation
            y = self.fc(y).view(b, c, 1, 1)  # [B, C] -> [B, C, 1, 1]
            # Recalibración: multiplicación de canales por sus pesos
            return x * y.expand_as(x)  # Broadcast y multiplicar

    class SEEfficientNet(nn.Module):
        def __init__(self, num_classes=1):
            super(SEEfficientNet, self).__init__()
            # Cargar modelo base preentrenado
            self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            
            # Obtener la capa de características del EfficientNet
            self.features = self.base_model.features
            
            # Añadir bloques SE a cada bloque de características
            for i in range(len(self.features)):
                # Obtenemos el número de canales del bloque actual
                if hasattr(self.features[i], 'block'):
                    # Si es un MBConvBlock, añadir SE a nivel de bloque
                    channels = self.features[i]._blocks[-1].project_conv.out_channels
                    # Insertar un bloque SE después de cada MBConvBlock
                    se_block = SEBlock(channels)
                    # Guardamos el bloque SE en el módulo para que pueda ser accedido durante forward
                    setattr(self, f'se_block_{i}', se_block)
            
            # Capa de clasificación más compleja con dropout y capas intermedias
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),  # Dropout para regularización
                nn.Linear(in_features=1280, out_features=512),
                nn.BatchNorm1d(512),  # Batch normalization
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(in_features=512, out_features=128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(in_features=128, out_features=num_classes)  # Salida final (binaria)
            )
        
        def forward(self, x):
            # Pasar por cada bloque de features y aplicar SE donde corresponda
            for i in range(len(self.features)):
                # Aplicar el bloque convolucional
                x = self.features[i](x)
                # Si tiene un bloque SE asociado, aplicarlo
                if hasattr(self, f'se_block_{i}'):
                    se_block = getattr(self, f'se_block_{i}')
                    x = se_block(x)
            
            # Global average pooling y flatten
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            
            # Clasificador
            return self.classifier(x)

    class EnsembleModel(nn.Module):
        def __init__(self, vit_model, efficientnet_model, vit_weight=0.6, efficientnet_weight=0.4):
            super(EnsembleModel, self).__init__()
            self.vit_model = vit_model
            self.efficientnet_model = efficientnet_model
            self.vit_weight = vit_weight
            self.efficientnet_weight = efficientnet_weight
            
        def forward(self, x):
            # Obtenemos predicciones de cada modelo
            vit_output = self.vit_model(x)
            efficientnet_output = self.efficientnet_model(x)
            
            # Verificamos y ajustamos dimensiones
            # Para clasificación binaria, aseguramos que ambos modelos tengan la misma forma
            if vit_output.shape[1] > 1 and efficientnet_output.shape[1] == 1:
                # Si ViT da [batch_size, 2] y EfficientNet da [batch_size, 1]
                # Convertimos EfficientNet a formato ViT
                eff_sigmoid = torch.sigmoid(efficientnet_output).view(-1, 1)
                efficientnet_probs = torch.cat([1 - eff_sigmoid, eff_sigmoid], dim=1)
                
                # Usamos softmax para ViT
                vit_probs = torch.softmax(vit_output, dim=1)
                
            elif vit_output.shape[1] == 1 and efficientnet_output.shape[1] == 1:
                # Ambos son binarios con sigmoid
                vit_probs = torch.sigmoid(vit_output)
                efficientnet_probs = torch.sigmoid(efficientnet_output)
                
            else:
                # Multiclase: usamos softmax para ambos
                vit_probs = torch.softmax(vit_output, dim=1)
                efficientnet_probs = torch.softmax(efficientnet_output, dim=1)
            
            # Combinamos las predicciones con pesos
            ensemble_output = (self.vit_weight * vit_probs + 
                              self.efficientnet_weight * efficientnet_probs)
            
            return ensemble_output
        
        def get_individual_predictions(self, x):
            """Retorna las predicciones individuales de cada modelo para análisis"""
            vit_output = self.vit_model(x)
            efficientnet_output = self.efficientnet_model(x)
            
            return vit_output, efficientnet_output

    def load_pretrained_models(num_classes=2):
        # Cargar modelo ViT
        vit_model = create_vit_classifier(num_classes=num_classes)
        
        # Cargar estado del modelo ViT
        vit_checkpoint = torch.load(VIT_MODEL_PATH, map_location=device)
        # Comprobar si es un checkpoint completo o solo el state_dict
        if 'model_state_dict' in vit_checkpoint:
            vit_model.load_state_dict(vit_checkpoint['model_state_dict'])
        else:
            vit_model.load_state_dict(vit_checkpoint)
        
        vit_model = vit_model.to(device)
        vit_model.eval()  # Poner en modo evaluación
        
        # Cargar modelo EfficientNet
        if num_classes == 2:
            # Para clasificación binaria, EfficientNet usa 1 salida
            efficientnet_model = SEEfficientNet(num_classes=1)
        else:
            efficientnet_model = SEEfficientNet(num_classes=num_classes)
        
        # Cargar estado del modelo EfficientNet
        eff_checkpoint = torch.load(EFFICIENT_MODEL_PATH, map_location=device)
        
        # Comprobar si es un checkpoint completo o solo el state_dict
        if 'model_state_dict' in eff_checkpoint:
            efficientnet_model.load_state_dict(eff_checkpoint['model_state_dict'])
        else:
            efficientnet_model.load_state_dict(eff_checkpoint)
        
        efficientnet_model = efficientnet_model.to(device)
        efficientnet_model.eval()  # Poner en modo evaluación
        
        return vit_model, efficientnet_model
    
    try:
        # Cargar los modelos
        vit_model, efficientnet_model = load_pretrained_models(num_classes)
        print("Modelos cargados exitosamente!")
        
        # Optimizar los pesos usando metaheurísticas
        print("\nIniciando optimización de pesos con métodos metaheurísticos...")
        optimized_ensemble, best_method, best_weights, best_fitness = optimize_ensemble_weights(
            test_loader, vit_model, efficientnet_model, device, num_classes
        )
        
        # Evaluar el ensemble optimizado con el dataset completo
        print("\nEvaluando el modelo ensamblado optimizado...")
        from sklearn.metrics import confusion_matrix, classification_report, f1_score
        
        def evaluate_ensemble(ensemble_model, test_loader, num_classes=2):
            ensemble_model.eval()
            all_preds = []
            all_true = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Predicciones del ensemble
                    outputs = ensemble_model(inputs)
                    
                    # Procesar predicciones
                    if num_classes == 2:
                        if outputs.shape[1] > 1:
                            preds = (outputs[:, 1] > 0.5).float()
                        else:
                            preds = (outputs.view(-1) > 0.5).float()
                    else:
                        preds = torch.argmax(outputs, dim=1)
                    
                    # Asegurar forma correcta
                    preds = preds.view(-1)
                    labels = labels.view(-1)
                    
                    # Guardar predicciones y etiquetas
                    all_preds.extend(preds.cpu().numpy())
                    all_true.extend(labels.cpu().numpy())
            
            # Calcular métricas
            cm = confusion_matrix(all_true, all_preds)
            report = classification_report(all_true, all_preds, output_dict=True)
            f1 = f1_score(all_true, all_preds, average='weighted')
            
            return cm, report, f1
        
        cm, report, f1 = evaluate_ensemble(optimized_ensemble, test_loader, num_classes)
        
        # Mostrar matriz de confusión
        print("\nMatriz de Confusión:")
        print(cm)
        
        # Mostrar reporte detallado
        print("\nInforme de Clasificación:")
        for cls in report:
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"Clase {cls}: Precision={report[cls]['precision']:.4f}, "
                      f"Recall={report[cls]['recall']:.4f}, "
                      f"F1={report[cls]['f1-score']:.4f}")
        
        print(f"\nAccuracy: {report['accuracy']:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        
        # Visualizar matriz de confusión
        import seaborn as sns
        
        def plot_confusion_matrix(cm, class_names):
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix - Optimized Ensemble')
            plt.tight_layout()
            plt.savefig('confusion_matrix_optimized_ensemble.png')
            plt.show()
        
        if num_classes == 2:
            class_names = ['Clase 0', 'Clase 1']
        else:
            class_names = [f'Clase {i}' for i in range(num_classes)]
        
        plot_confusion_matrix(cm, class_names)
        
        # Guardar el modelo optimizado
        torch.save({
            'model_state_dict': optimized_ensemble.state_dict(),
            'best_method': best_method,
            'best_weights': best_weights,
            'best_fitness': best_fitness
        }, 'optimized_ensemble_model.pth')
        
        print(f"\nModelo optimizado guardado como 'optimized_ensemble_model.pth'")
        print(f"Método: {best_method}")
        print(f"Pesos: ViT={best_weights[0]:.4f}, EfficientNet={best_weights[1]:.4f}")
        print(f"F1 Score: {best_fitness:.4f}")
        
    except Exception as e:
        import traceback
        print(f"Error durante la ejecución: {e}")
        traceback.print_exc()