#!/usr/bin/env python3

from deap import base, creator, tools
from deap import algorithms
import numpy
import itertools
import random
import sys
import matplotlib.pyplot as plt
import collections

# 
# Evolution Constansts and Variables
# 

N_POPULATION = 1000
N_GENERATIONS = 100
REBALANCING = 20
IMPORTANT_RAM_SIZE = 1024
IMPORTANT_LOAD = 1
OVERLOAD_COEF = 1.3

#METRICS = ["min", "max", "avg"]
METRICS = ["min"]

RESOURCES_NAMES = [ "load", "ram", "disk", "net" ]
#RESOURCES_NAMES = [ "load", "ram", ]

LIMIT_TIMES = ['1608660000', '1609102800', '1608562800']

RESOURCES = []

# Tournament Selection
TOURNSIZE = 3

# Mating Probabilities
CXPB = 0.4

RESET_SMALL_GENES_PB = 0.01

# Mutations Probabilities
MUTPB = 0.2

MUTATE_GENE_PB = 0.005
RESET_IND_PB = 0.008
SWAP_BASE_PB = 0.005

NUKE_BOMB_PB = 0.001
NUKE_BOMB_EFFECT = 0.1

MIRACLE_PB = 0.005
MIRACLE_EFFECT = 0.1

# 
# Global variables
# 
N_VIRTUAL = 0


# NUKE_BOMB & MIRACLE
nuke_fired = False
nuke_counter = 0

miracle_happened = False
miracle_counter = 0

class Hypervisor:
    def __init__(self):
        self.resources = dict()
        for i in RESOURCES:
            self.resources[i] = 0

    def str_values(self):
        od = collections.OrderedDict(sorted(self.resources.items()))
        o = '\t'.join( "%s / %0.2f" % (k, float(v)) for k,v in od.items() )
        return o
        #return "%d\t%0.2f\t%0.2f\t%0.2f" % (self.ram, self.load, self.disk, self.net)

class Virtual:
    def __init__(self, name):
        self.name = name

        self.resources = dict()
        for i in RESOURCES:
            self.resources[i] = 0
    
    def __repr__(self):
        #return str(self.__dict__)
        return "<Virtual: %s>" % self.name

vm_map = dict()
vm_cnt = 0

hv_map = dict()
hv_cnt = 0

def get_hv_id(name):
    global hv_map, hv_cnt
    if name not in hv_map.keys():
        hv_map[name] = hv_cnt
        hv_cnt += 1
    
    return hv_map[name]

def get_hvname(id):
    for key, val in hv_map.items():
        if val == id:
            return key
    raise 

adam = list()
with open("samples/adam.txt", "r") as f:
    data = f.read().replace('\r', '').split('\n')
    for row in data:
        hv, vmid = row.split(' ', 1)
        hvid = get_hv_id(hv)
        vm_map[vm_cnt] = int(vmid)
        adam.append(hvid)
        vm_cnt += 1

N_HYPERVISOR = hv_cnt
N_VIRTUAL = vm_cnt

print("N_HYPERVISOR:", hv_cnt)
print("N_VIRTUAL:", vm_cnt)

def get_vm_pos(vmid):
    if type(vmid) == str:
        vmid = int(vmid)
    for key, val in vm_map.items():
        if val == vmid:
            return key
    raise 

virtuals = [ None, ] * N_VIRTUAL

timepoints = set()
with open("samples/data.txt", "r") as f:
    data = f.read().replace('\r', '').split('\n')
    for row in data:
        row = row.split(' ')
        timepoints.add(row[0])

print(timepoints)

for r in RESOURCES_NAMES:
    for t in timepoints:
        if t not in LIMIT_TIMES:
            continue
        rr = r + "_" + t
        RESOURCES.append(rr)

with open("samples/data.txt", "r") as f:
    data = f.read().replace('\r', '').split('\n')
    for row in data:
        #if not row: continue
        row = row.split(' ')
        tp = row[0]
        if tp not in LIMIT_TIMES:
            continue
        vmid = row[1]
        resources = row[2:]
        pos = get_vm_pos(vmid)

        if virtuals[pos] == None:
            virtuals[pos] = Virtual(vmid)

        for i in range(len(RESOURCES_NAMES)):
            r = RESOURCES_NAMES[i] + "_" + tp
            virtuals[pos].resources[r] = float(resources[i])


for i in range(len(virtuals)):
    if virtuals[i] == None:
        vm_name = vm_map[i]
        print("No data for", vm_name)
        virtuals[i] = Virtual(vm_name)

hvall = Hypervisor()
for i in range(N_VIRTUAL):
    for r in RESOURCES:
        hvall.resources[r] += virtuals[i].resources[r]

print("Cluster summary:")

for k, v in hvall.resources.items():
    vv = hvall.resources[k]
    print(" %s sum:\t%0.2f" % (k, round(vv)))
    print(" %s avg:\t%0.2f" % (k, round(vv / N_HYPERVISOR)))

print()

def random_hv():
    return random.randint(0, N_HYPERVISOR-1)

def calc_changes(ind):
    n = 0
    for i in range(N_VIRTUAL):
        if ind[i] != adam[i]:
            n += 1
    return n


def evaluate(ind, verbose=False):
    hv = list()
    for i in range(N_HYPERVISOR):
        hv.append(Hypervisor())
    
    for i in range(N_VIRTUAL):
        parent_hv = ind[i]

        for j in RESOURCES:
            hv[parent_hv].resources[j] += virtuals[i].resources[j]

    #for i in hv:
    #    print(i.resources)

    decr = 0
    for i in range(N_HYPERVISOR):
        for r in RESOURCES:
            rr = r.split('_')[0]
            if rr in ['ram', 'load']:
                if rr == 'ram':
                    if hv[i].resources[r] > (hvall.resources[r] / N_HYPERVISOR * OVERLOAD_COEF):
                        decr += 100
                if rr == 'load': 
                    if hv[i].resources[r] > (hvall.resources[r] / N_HYPERVISOR * OVERLOAD_COEF):
                        decr += 100

    perm = list(itertools.product(hv, hv))

    # find worst ratio between each two combinations
    # it will be our fitness per resource
    maxims = {}
    for r in RESOURCES:
        m = 0
        for x, y in perm:
            if x.resources[r] <= y.resources[r]:
                continue
            if y.resources[r] == 0:
                continue
            rr = x.resources[r]/y.resources[r] - 1
            if rr > m:
                m = rr
        maxims[r] = round(m, 2)

    # fitness changes
    changes = calc_changes(ind)
    if changes == 0:
        changes = N_VIRTUAL

    fitness_changes = round(2 ** (changes/REBALANCING) -1, 2)

    fit_value = round(sum(maxims.values()) + decr, 2)

    if verbose:
        print("Cluster balance:")
        for i in range(len(hv)):
            print(" ", get_hvname(i), hv[i].str_values())
        
        print("Changes:", changes)
        print("Fitness:")
        print(" penalization: %d" % decr)
        print(" ftns_resources: %0.2f" % fit_value)
        print(" ftns_changes: %0.2f" % fitness_changes)
        for r in RESOURCES:
            print("  ftns_%s: %0.2f" % (r, maxims[r]))
    
    return fit_value, fitness_changes


def _mutRandom(mutant, pb):
    for i in range(N_VIRTUAL):
        if random.uniform(0, 1) < pb:
            mutant[i] = random_hv()


def _mutRandomize(mutant):
    return _mutRandom(mutant, 1)

def _triggerMiracle(mutant):
    global miracle_happened
    miracle_happened = True
    #print("Miracle!")
    _mutHeal(mutant)

def _mutHeal(mutant):
    return mutResetUnimportantGenes(mutant)

def _handleMiracle(mutant):
    global miracle_happened, miracle_counter

    miracle_counter += 1
    if miracle_counter == round(N_POPULATION * MIRACLE_EFFECT):
        miracle_counter = 0
        miracle_happened = False
    
    return _mutHeal(mutant)


def _nukeFire(mutant):
    global nuke_fired
    nuke_fired = True
    #print("Nuke!")
    _mutRandomize(mutant)


def _handleNuke(mutant):
    global nuke_fired, nuke_counter

    nuke_counter += 1
    if nuke_counter == round(N_POPULATION * NUKE_BOMB_EFFECT):
        nuke_counter = 0
        nuke_fired = False

    _mutNukeBomb(mutant)


def _mutNukeBomb(mutant):
    return _mutRandom(mutant, 0.8)
    #return mutResetUnimportantGenes(mutant)


def _mutSwapBase(mutant, pb=SWAP_BASE_PB, rounds=1):
    for i in range(rounds):
        if random.uniform(0, 1) < pb:
            x, y = random_hv(), random_hv()
            if x != y:
                for i in range(N_VIRTUAL):
                    if mutant[i] == x:
                        mutant[i] = y
                    elif mutant[i] == y:
                        mutant[i] = x


def mutResetUnimportantGenes(ind):
    #print("mutResetUnimportantGenes")
    mutant = toolbox.clone(ind)

    for i in range(N_VIRTUAL):
        if adam[i] == mutant[i]:
            continue
        
        #if virtuals[i].ram < IMPORTANT_RAM_SIZE and virtuals[i].load < IMPORTANT_LOAD:
        #    mutant[i] = adam[i]
    
    del mutant.fitness.values
    return mutant


def mutate(individual):
    mutant = toolbox.clone(individual)

    _mutRandom(mutant, pb=MUTATE_GENE_PB)

    if random.uniform(0, 1) < RESET_IND_PB:
        _mutRandomize(mutant)

    if random.uniform(0, 1) < NUKE_BOMB_PB:
        _nukeFire(mutant)

    if random.uniform(0, 1) < MIRACLE_PB:
        _triggerMiracle(mutant)

    if nuke_fired:
        _handleNuke(mutant)

    if miracle_happened:
        mutant = _handleMiracle(mutant)
    
    _mutSwapBase(mutant, pb=SWAP_BASE_PB, rounds=1)

    del mutant.fitness.values
    return mutant,


def selTournamentMultiFitness(individuals, k, tournsize):
    chosen = []
    for i in range(k):
        aspirants = tools.selRandom(individuals, tournsize)
        asp = aspirants[0]
        m = sum(asp.fitness.values)
        for a in aspirants:
            fit = sum(a.fitness.values)
            if m > fit:
                fit = m
                asp = a
        chosen.append(asp)
    return chosen


def mutCloneWithSmallGenesResetPb(individual, pb=RESET_SMALL_GENES_PB):
    if random.uniform(0, 1) < pb:
        return mutResetUnimportantGenes(individual)

    clone = toolbox.clone(individual)
    del clone.fitness.values
    return clone


def cxTwoPointWithSmallGenesReset(ind1, ind2):
    i1 = mutCloneWithSmallGenesResetPb(ind1)
    i2 = mutCloneWithSmallGenesResetPb(ind2)

    i1, i2 = tools.cxTwoPoint(i1, i2)

    del i1.fitness.values
    del i2.fitness.values

    return i1, i2


# 
# Statistics
# 

fitness_map = [
    "ftns_resources",
    "ftns_changes",
]

stats_args = {}

for i in range(len(fitness_map)):
    k = fitness_map[i]
    stats_args[k] = tools.Statistics(key=lambda ind, x=i: ind.fitness.values[x])

mstats = tools.MultiStatistics(**stats_args)

mstats.register("min", numpy.min)
mstats.register("max", numpy.max)
mstats.register("avg", numpy.average)

def adams_child():
    child = list(adam)
    _mutRandom(child, 0.01)
    return creator.Individual(child)


#
# Evolution
#
#evaluate(adam, True)

creator.create("FitnessMulti", base.Fitness, weights=(-1.0,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("individual", adams_child)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", cxTwoPointWithSmallGenesReset)
toolbox.register("mutate", mutate)

toolbox.register("select", selTournamentMultiFitness, tournsize=TOURNSIZE)
toolbox.register("evaluate", evaluate)

pop = toolbox.population(n=N_POPULATION)
hof = tools.HallOfFame(10)

pop, logbook = algorithms.eaSimple(
    pop, toolbox,
    cxpb=CXPB, mutpb=MUTPB, ngen=N_GENERATIONS, 
    stats=mstats, verbose=True,
    halloffame=hof)


# 
# Output
# 

print("==========================================")
print("Adam:")
evaluate(adam, True)
print()

print("==========================================")
print("============= HALL OF FAME ===============")
print("==========================================")
print()

j = 0
for ind in hof:
    print("==========================================")
    print("=== %d === " % j)
    print()
    j += 1

    print("#%d - %d changes, fitness %0.2f" % (i, calc_changes(ind), ind.fitness.values[0]) )

    evaluate(ind, verbose=True)
    print()
    for i in range(N_VIRTUAL):
        if ind[i] != adam[i]:
            old_hv = get_hvname(adam[i])
            new_hv = get_hvname(ind[i])
            vm_name = vm_map[i]
            print("Rebalance: vmid %s from %s to %s" % (vm_name, old_hv, new_hv))

    print()

gen = logbook.select("gen")
fig, ax1 = plt.subplots()
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness")

lines = {}


for metric in METRICS:
    for key in logbook.chapters.keys():
        data = logbook.chapters[key].select(metric)
        c = numpy.random.rand(3,)
        l = ax1.plot(gen, data, 0, label=str(key + "_" + metric))
        lines[key + "_" + metric] = l[0]


#fitness_avg = [ 0, ] * len(gen)
#for metric in ['min',]:
#    for key in logbook.chapters.keys():
#        data = logbook.chapters[key].select(metric)
#        fitness_avg = [ (x+y)/2.0 for x, y in zip(fitness_avg, data)]
#
#c = numpy.random.rand(3,)
#l = ax1.plot(gen, fitness_avg, 0, label="fitness")
#lines["fitness"] = l[0]

lns = [ v for k,v in lines.items() ]
labs = [ x.get_label() for x in lns ]
ax1.legend(lns, labs, loc="best")

plt.show()
