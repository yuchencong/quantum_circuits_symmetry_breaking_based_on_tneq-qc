import numpy as np
import random
# from mpi_core import DUMMYINDV

class FITNESS_FUNCS:

    @staticmethod
    def defualt(s, l):
        return s + l * 50

class EVOLVE_OPS:

    @staticmethod
    def fillup(generation, adj_func):
        self.societies[society_name]['indv'] = [ \
                        Individual(scope='{}/{}/{:03d}'.format(self.name, society_name, i), 
                        adj_func=Individual.naive_random_adj_matrix_with_sparsity_limitation) \
                        for i in range(self.kwds['population'][n]) ]


    @staticmethod
    def mutation(indv, prob, generation):
        dim = indv.adj_matrix.shape[0]
        elements = np.stack(np.triu_indices(dim, 1)).transpose()
        mask = np.random.uniform(size=elements.shape[0])<prob
        mutated_elements = tuple(map(tuple, elements[mask].transpose()))
        if mutated_elements:
            indv.adj_matrix[mutated_elements] = generation.rank - indv.adj_matrix[mutated_elements]
            indv.adj_matrix[np.tril_indices(dim, -1)] = indv.adj_matrix.transpose()[np.tril_indices(dim, -1)]

    @staticmethod
    def immigration(societies, number=5):
        society_A, society_B = societies
        for _ in range(number):
            society_B.append(society_A.pop(0))
            society_A.append(society_B.pop(0))

    @staticmethod
    def elimination(society, threshold=80):
        society['rank'] = society['rank'][:threshold]
        society['indv'] = [society['indv'][i] for i in society['rank']]
        society['total'] = [society['total'][i] for i in society['rank']]

    @staticmethod
    def crossover(society, population, alpha=5):
        __adj_matrix__, __parents__ = [], []
        def propagation(couple, percent=0.5):
            adj_matrix_male = np.copy(couple[0].adj_matrix)
            adj_matrix_female = np.copy(couple[1].adj_matrix)

            dim = adj_matrix_male.shape[0]
            exchange_core = random.choice(list(range(dim)))

            exchange = adj_matrix_male[exchange_core]
            adj_matrix_male[exchange_core] = adj_matrix_female[exchange_core] 
            adj_matrix_female[exchange_core] = exchange

            adj_matrix_male[np.tril_indices(dim, -1)] = adj_matrix_male.transpose()[np.tril_indices(dim, -1)]
            adj_matrix_female[np.tril_indices(dim, -1)] = adj_matrix_female.transpose()[np.tril_indices(dim, -1)]

            __adj_matrix__.append(adj_matrix_male)
            __adj_matrix__.append(adj_matrix_female)
            __parents__.append((couple[0].scope[-13:], couple[1].scope[-13:]))
            __parents__.append((couple[0].scope[-13:], couple[1].scope[-13:]))

        indv, fitness = society['indv'], society['total']
        rank = np.argsort(fitness)
        # prob = [ 1.0/(1e-5+f)*alpha for f in fitness]        
        # p = [ np.exp(3/(1+k)) for k in range(len(indv)) ]
        p = [ np.maximum(np.log(float(sys.argv[4])/(0.01+k*5)), 0.01) for k in range(population) ]
        prob = np.zeros(len(indv))
        for idx, i in enumerate(rank): prob[i] = p[idx]
        for i in range(population//2): propagation(random.choices(indv, weights=prob, k=2))
        for i in range(population-len(indv)): indv.append(DUMMYINDV())
        for v, m, p in zip(indv, __adj_matrix__, __parents__): v.adj_matrix, v.parents = m, p