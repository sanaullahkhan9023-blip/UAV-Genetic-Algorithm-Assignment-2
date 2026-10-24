"""Multi-objective UAV mission strategy engine (educational hazard-grid simulation).
All distance/time/energy/risk quantities are simplified simulation proxies, not aircraft specifications.
"""
from dataclasses import dataclass
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRID = 30
DRONES = 3
STOPS = 4
BASE = np.array([1.0, 1.0])
MAX_ROUTE = 105.0
MAX_TIME = 75.0
# Artificial restricted-airspace rectangles (inclusive bounds).
NO_FLY = [(12, 17, 10, 18), (4, 8, 22, 26)]
WEIGHTS = {'coverage': .32, 'detection': .20, 'energy': .17, 'time': .13, 'risk': .18}


def inside_restricted(p):
    x, y = p
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0,x1,y0,y1 in NO_FLY)


def segment_samples(a, b, resolution=.5):
    n = max(2, int(np.ceil(np.linalg.norm(np.asarray(b)-np.asarray(a)) / resolution)) + 1)
    return np.linspace(a,b,n)


def segment_restricted(a,b):
    # Fast exact segment-vs-rectangle intersection via Liang-Barsky clipping.
    ax,ay=float(a[0]),float(a[1]); dx=float(b[0]-a[0]); dy=float(b[1]-a[1])
    for x0,x1,y0,y1 in NO_FLY:
        lo,hi=0.,1.
        for p,q in [(-dx,ax-x0),(dx,x1-ax),(-dy,ay-y0),(dy,y1-ay)]:
            if abs(p)<1e-12:
                if q<0:lo,hi=1.,0.;break
            else:
                r=q/p
                if p<0:lo=max(lo,r)
                else:hi=min(hi,r)
        if lo<=hi:return True
    return False


FREE_CELLS=np.array([(x,y) for x in range(GRID) for y in range(GRID) if not inside_restricted((x,y))])

def repair_point(p, rng):
    p = np.clip(np.asarray(p, dtype=float), 0.0, GRID-1.)
    if not inside_restricted(p):
        return p
    # Deterministically snap to nearest free cell center.
    d = np.sum((FREE_CELLS-p)**2,axis=1)
    return FREE_CELLS[np.argmin(d)].astype(float)


def repair_segment(a,b,rng):
    """If direct segment hits obstacle, try axis-aligned free-corridor detours.

    Return intermediate path vertices. Fallback is a waypoint with a large safety penalty
    in evaluation rather than incorrectly claiming feasibility.
    """
    if not segment_restricted(a,b):
        return []
    candidates = []
    for x in [0.0, 2.0, 9.0, 10.0, 19.0, 20.0, 28.0, 29.0]:
        pts=[np.array([x,a[1]]), np.array([x,b[1]])]
        if all(not segment_restricted(u,v) for u,v in zip([a]+pts,pts+[b])):
            candidates.append(pts)
    for y in [0.0, 2.0, 8.0, 9.0, 19.0, 20.0, 28.0, 29.0]:
        pts=[np.array([a[0],y]), np.array([b[0],y])]
        if all(not segment_restricted(u,v) for u,v in zip([a]+pts,pts+[b])):
            candidates.append(pts)
    if candidates:
        return min(candidates,key=lambda pts: sum(np.linalg.norm(v-u) for u,v in zip([a]+pts,pts+[b])))
    return []


@dataclass
class Chromosome:
    """Hybrid representation: XY waypoints (float), speeds (float), sensors (integer 0/1/2).

    Each gene index d,k codes the kth stop visited by drone d, its outgoing speed,
    and its sensor setting (off/low/high) during the associated scan.
    """
    xy: np.ndarray
    speed: np.ndarray
    sensor: np.ndarray
    fitness: float = float('-inf')
    metrics: dict = None

    def clone(self):
        return Chromosome(self.xy.copy(),self.speed.copy(),self.sensor.copy(),self.fitness,
                          None if self.metrics is None else self.metrics.copy())


def initialize(rng):
    xy = rng.uniform(0, GRID-1, size=(DRONES,STOPS,2))
    speed = rng.uniform(.8,2.0,size=(DRONES,STOPS))
    sensor = rng.integers(0,3,size=(DRONES,STOPS))
    result=Chromosome(xy,speed,sensor)
    return repair(result,rng)


def repair(c,rng):
    c.xy=np.clip(c.xy,0,GRID-1)
    for d in range(DRONES):
        for k in range(STOPS):
            c.xy[d,k]=repair_point(c.xy[d,k],rng)
    c.speed=np.clip(c.speed,.8,2.)
    c.sensor=np.clip(np.rint(c.sensor),0,2).astype(int)
    c.fitness=float('-inf')
    c.metrics=None
    return c


def environment(seed=12):
    x,y=np.meshgrid(np.arange(GRID),np.arange(GRID),indexing='xy')
    priority=(.15 + 1.6*np.exp(-((x-8)**2+(y-11)**2)/36)
              +2.1*np.exp(-((x-24)**2+(y-23)**2)/42)
              +1.1*np.exp(-((x-22)**2+(y-5)**2)/35))
    hazard=(.06 + 1.1*np.exp(-((x-21)**2+(y-14)**2)/25)
            + .9*np.exp(-((x-7)**2+(y-23)**2)/30))
    blocked=np.zeros((GRID,GRID),bool)
    for x0,x1,y0,y1 in NO_FLY:
        blocked[y0:y1+1,x0:x1+1]=True
    priority[blocked]=0
    return priority,hazard,blocked


def routed_legs(c,d,rng):
    nodes=[BASE]+[c.xy[d,k] for k in range(STOPS)]+[BASE]
    path=[np.asarray(nodes[0])]
    unresolved=0
    for a,b in zip(nodes[:-1],nodes[1:]):
        via=repair_segment(a,b,rng)
        segment=[a]+via+[b]
        if any(segment_restricted(u,v) for u,v in zip(segment[:-1],segment[1:])):
            unresolved+=1
        path.extend(via+[b])
    return np.array(path),unresolved


def evaluate(c,priority,hazard,blocked,rng):
    """Weighted normalized objectives with exact-infeasibility penalties.

    Maps record unique-cell coverage (no double counting). Detection is a
    probability proxy, derived from sensor power and scanning distance.
    """
    heat=np.zeros((GRID,GRID),float)
    detection=np.zeros((GRID,GRID),float)
    total_energy=total_time=total_risk=total_distance=0.
    violations=0
    paths=[]
    for d in range(DRONES):
        path,unresolved=routed_legs(c,d,rng)
        paths.append(path)
        violations+=unresolved
        route_distance=0.
        # Evaluate each routed leg using the most recent stop's speed.
        for j,(a,b) in enumerate(zip(path[:-1],path[1:])):
            length=float(np.linalg.norm(b-a))
            speed=float(c.speed[d,min(j,STOPS-1)])
            route_distance+=length
            total_time+=length/speed
            total_energy+=length*(.7+.18*speed*speed)
            points=segment_samples(a,b)
            px=np.clip(np.rint(points[:,0]).astype(int),0,GRID-1)
            py=np.clip(np.rint(points[:,1]).astype(int),0,GRID-1)
            total_risk+=length*float(np.mean(hazard[py,px]))
            if np.any(blocked[py,px]): violations+=1
        total_distance+=route_distance
        if route_distance > MAX_ROUTE:
            violations+=(route_distance-MAX_ROUTE)/10
        # Scan circular footprints around each stop. 0=off, 1=low, 2=high.
        for k in range(STOPS):
            power=int(c.sensor[d,k]); center=c.xy[d,k]
            total_energy+=power*2.0
            total_time+=power*.45
            if power==0:continue
            radius=2.4 if power==1 else 4.0
            strength=.55 if power==1 else .87
            x,y=np.meshgrid(np.arange(GRID),np.arange(GRID))
            dist=np.sqrt((x-center[0])**2+(y-center[1])**2)
            scan=np.where((dist<=radius)&(~blocked), strength*np.exp(-dist*dist/(2*radius*radius)),0.)
            heat=np.maximum(heat,scan)
            detection=1-(1-detection)*(1-scan) # independent detection attempts
    if total_time > DRONES*MAX_TIME: violations+=(total_time-DRONES*MAX_TIME)/15
    priority_total=max(float(priority.sum()),1e-12)
    coverage=float((priority*(heat>0)).sum()/priority_total)
    detection_score=float((priority*detection).sum()/priority_total)
    energy_score=min(1.,total_energy/450.)
    time_score=min(1.,total_time/240.)
    risk_score=min(1.,total_risk/130.)
    objective=(WEIGHTS['coverage']*coverage + WEIGHTS['detection']*detection_score
               -WEIGHTS['energy']*energy_score-WEIGHTS['time']*time_score
               -WEIGHTS['risk']*risk_score)
    # Huge hard-constraint penalty ensures feasible candidates dominate infeasible ones.
    fitness=objective-2.0*violations
    c.fitness=float(fitness)
    c.metrics=dict(coverage=coverage,detection=detection_score,energy=total_energy,
                   time=total_time,risk=total_risk,distance=total_distance,
                   violations=violations,objective=objective)
    return heat,paths


def tournament(pop,rng,k=3):
    selected=rng.choice(len(pop),size=min(k,len(pop)),replace=False)
    return max((pop[int(i)] for i in selected),key=lambda v:v.fitness)


def crossover(a,b,rng):
    """Swap whole drone segments, then blend a random waypoint coordinate pair."""
    c=a.clone()
    idx=int(rng.integers(DRONES))
    c.xy[idx]=b.xy[idx].copy()
    c.speed[idx]=b.speed[idx].copy()
    c.sensor[idx]=b.sensor[idx].copy()
    d,k=int(rng.integers(DRONES)),int(rng.integers(STOPS))
    alpha=rng.uniform(.15,.85)
    c.xy[d,k]=alpha*a.xy[d,k]+(1-alpha)*b.xy[d,k]
    return c


def mutate(c,rng,rate):
    """Gaussian local shifts + occasional global relocation and mixed-gene mutation."""
    for d in range(DRONES):
        for k in range(STOPS):
            if rng.random()<rate:
                c.xy[d,k]+=rng.normal(0,2.8,size=2) if rng.random()<.85 else rng.uniform(0,GRID-1,size=2)-c.xy[d,k]
            if rng.random()<rate: c.speed[d,k]+=rng.normal(0,.25)
            if rng.random()<rate: c.sensor[d,k]=int(rng.integers(0,3))
    return repair(c,rng)


def run_ga(priority,hazard,blocked,seed=12,popsize=32,generations=45,mutation=.14,tournament_size=3):
    rng=np.random.default_rng(seed)
    pop=[initialize(rng) for _ in range(popsize)]
    for c in pop:evaluate(c,priority,hazard,blocked,rng)
    history=[]
    champions=[]
    best=max(pop,key=lambda c:c.fitness).clone()
    for generation in range(generations+1):
        generation_best=max(pop,key=lambda c:c.fitness)
        if generation_best.fitness>best.fitness:best=generation_best.clone()
        history.append((generation,best.fitness,float(np.mean([i.fitness for i in pop]))))
        if generation in [0,generations//2,generations]:champions.append((generation,best.clone()))
        if generation==generations:break
        nxt=[best.clone(),generation_best.clone()]  # elitism
        while len(nxt)<popsize:
            p=tournament(pop,rng,tournament_size)
            q=tournament(pop,rng,tournament_size)
            child=crossover(p,q,rng) if rng.random()<.85 else p.clone()
            mutate(child,rng,mutation)
            evaluate(child,priority,hazard,blocked,rng)
            nxt.append(child)
        pop=nxt
    return best,np.asarray(history),champions


def plot_solution(c,priority,hazard,blocked,path,rng,title):
    heat,paths=evaluate(c,priority,hazard,blocked,rng)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axes[0].imshow(priority,origin='lower',cmap='YlOrRd')
    axes[0].contour(blocked.astype(int),levels=[.5],colors='black',linewidths=1.5)
    for i,p in enumerate(paths):
        axes[0].plot(p[:,0],p[:,1],'-o',markersize=2,label=f'UAV {i+1}')
    axes[0].plot(*BASE,marker='*',markersize=15,color='black',label='Base')
    axes[0].set_title('Priority grid and routes')
    axes[0].legend(loc='upper left',fontsize=8)
    im=axes[1].imshow(heat,origin='lower',vmin=0,vmax=.9,cmap='viridis')
    axes[1].contour(blocked.astype(int),levels=[.5],colors='white',linewidths=1.5)
    axes[1].set_title('Sensor coverage heatmap')
    fig.colorbar(im,ax=axes[1],shrink=.8,label='Maximum scan intensity')
    fig.suptitle(title)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def save_history(history,path):
    with open(path,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['generation','best_fitness','average_fitness']);w.writerows(history)
    fig,ax=plt.subplots(figsize=(8,4.7))
    ax.plot(history[:,0],history[:,1],label='Best-so-far fitness',lw=2)
    ax.plot(history[:,0],history[:,2],label='Average fitness',lw=1.8)
    ax.set(xlabel='Generation',ylabel='Penalized fitness',title='Genetic algorithm convergence')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(path.replace('.csv','.png'),dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description='UAV GA simulator')
    parser.add_argument('--output',default='outputs')
    parser.add_argument('--seed',type=int,default=12)
    parser.add_argument('--population',type=int,default=18)
    parser.add_argument('--generations',type=int,default=20)
    parser.add_argument('--mutation',type=float,default=.14)
    args=parser.parse_args()
    os.makedirs(args.output,exist_ok=True)
    priority,hazard,blocked=environment()
    best,hist,champions=run_ga(priority,hazard,blocked,args.seed,args.population,args.generations,args.mutation)
    save_history(hist,os.path.join(args.output,'fitness_history.csv'))
    for g,c in champions:
        plot_solution(c,priority,hazard,blocked,os.path.join(args.output,f'routes_generation_{g}.png'),np.random.default_rng(99),f'Best-so-far strategy - generation {g}')
    rng=np.random.default_rng(args.seed+1000)
    random_best=None
    for _ in range(args.population*(args.generations+1)):
        c=initialize(rng);evaluate(c,priority,hazard,blocked,rng)
        if random_best is None or c.fitness>random_best.fitness:random_best=c.clone()
    # Identical number of fitness evaluations makes budget comparison fair.
    with open(os.path.join(args.output,'summary.txt'),'w') as f:
        for name,c in [('GA',best),('Random search',random_best)]:
            line=f'{name}: fitness={c.fitness:.5f}; '+ '; '.join(f'{k}={v:.4f}' for k,v in c.metrics.items())
            print(line);f.write(line+'\n')
    plot_solution(random_best,priority,hazard,blocked,os.path.join(args.output,'random_search.png'),rng,'Random search baseline')
    with open(os.path.join(args.output,'sensitivity.csv'),'w',newline='') as f:
        w=csv.writer(f);w.writerow(['variable','setting','best_fitness','coverage','violations'])
        experiments=[('population',12,12, .14,3),('population',24,24,.14,3),
                     ('mutation',.05,18,.05,3),('mutation',.28,18,.28,3),
                     ('tournament',2,18,.14,2),('tournament',5,18,.14,5)]
        for label,setting,pop,rate,k in experiments:
            b,_,_=run_ga(priority,hazard,blocked,args.seed,pop,10,rate,k)
            w.writerow([label,setting,b.fitness,b.metrics['coverage'],b.metrics['violations']])
            print(f'Sensitivity {label}={setting}: fitness={b.fitness:.5f}')

if __name__=='__main__':main()
