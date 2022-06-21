import pygad
import datetime
from modeling import run
from pprint import pprint


class GATuner():
    def __init__(self, site_data
                 , site_scalar
                 , kwarg_names
                 , map_func
                 , n_pm25
                 , aqs_id=''
                 , log_ext=''
                 , num_generations=20
                 , sol_per_pop=100
                 , num_parents_mating=80
                 , keep_parents=20
                 , num_genes=7
                 , init_range_low=0.0001
                 , init_range_high=0.9999
                 , parent_selection_type="sss"
                 , crossover_type="single_point" # "two_points"
                 , mutation_type="random"
                 , mutation_percent_genes=30
                 , mutation_by_replacement=False
                 , random_mutation_min_val=0.0001
                 , random_mutation_max_val=0.9999
                 , save_best_solutions=True
                ):
        self.site_data = site_data
        self.scalar = site_scalar
        self.kwarg_names = kwarg_names
        self.map_func = map_func
        self.n_pm25 = n_pm25
        self.aqs_id = aqs_id
        self.log_ext = log_ext
        self.num_generations = num_generations
        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        self.keep_parents = keep_parents
        self.num_genes = num_genes
        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
        self.parent_selection_type = parent_selection_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.mutation_percent_genes = mutation_percent_genes
        self.mutation_by_replacement = mutation_by_replacement
        self.random_mutation_min_val = random_mutation_min_val
        self.random_mutation_max_val = random_mutation_max_val
        self.save_best_solutions = save_best_solutions
        self.g_iterations = 0
        self.g_r2 = -1
        self.g_fitness = []
        timestamp = datetime.datetime.timestamp(datetime.datetime.now())
        self.logFileStream = open(f'./logs/lasvegas_{self.aqs_id}_{self.log_ext}_log_{str(timestamp)}.txt', 'wt')

    def __del__(self):
        if self.logFileStream:
            self.logFileStream.close()

    def chromosome2kwargs(self, chrom):
        if len(chrom) != len(self.kwarg_names):
            raise Exception('Num of genes are not correct!')
        argv = [f(a) for f, a in zip(self.map_func, chrom)]
        kwargs = {key: val for key, val in zip(self.kwarg_names, argv)}
        return kwargs

    def tune(self):
        self.g_iterations = 0
        self.g_r2 = -1
        self.g_fitness = []
        def fitness_func(solution, solution_idx):
            self.g_iterations += 1
            print('='*100, self.g_iterations)
            print(str(datetime.datetime.now()))
            self.logFileStream.write('%s%s\n' % ('='*100, self.g_iterations))
            if len(solution ) != self.num_genes:
                raise Exception('Length of solution is not correct!')
            kwargs = self.chromosome2kwargs(solution)
            pprint(kwargs)
            self.logFileStream.write('%s\n' % (str(kwargs)))
            measures = run(self.site_data, self.scalar, self.n_pm25, plot=False, verbose=0, **kwargs)
            if measures['R^2'] > self.g_r2:
                self.g_r2 = measures['R^2']
                print('*'*50)
                pprint(measures)
                self.logFileStream.write('%s\n' % ('*'*50))
            self.logFileStream.write('%s\n' % (str(measures)))
            return measures['R^2']

        def callback_generation(ga_instance):
            ave_fitness = ga_instance.cal_pop_fitness().mean()
            self.g_fitness.append(ave_fitness)
            print('+'*10, 'Generation Average Fitness:', ave_fitness)

        def on_stop(ga_instance, last_population_fitness):
            self.logFileStream.write('=' * 100 + '\n')
            self.logFileStream.write('----- Average fitness along generations: \n')
            self.logFileStream.write('%s\n' % (str(self.g_fitness)))
            print('=' * 100, '\n')
            print('----- Average fitness along generations: \n')
            print('%s\n' % (str(self.g_fitness)))

        ga_instance = pygad.GA(num_generations=self.num_generations,
                       num_parents_mating=self.num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=self.sol_per_pop,
                       num_genes=self.num_genes,
                       init_range_low=self.init_range_low,
                       init_range_high=self.init_range_high,
                       parent_selection_type=self.parent_selection_type,
                       keep_parents=self.keep_parents,
                       crossover_type=self.crossover_type,
                       mutation_type=self.mutation_type,
                       mutation_by_replacement = self.mutation_by_replacement,
                       mutation_percent_genes=self.mutation_percent_genes,
                       random_mutation_min_val = self.random_mutation_min_val,
                       random_mutation_max_val = self.random_mutation_max_val,
                       save_best_solutions = self.save_best_solutions,
                       on_generation=callback_generation,
                       on_stop=on_stop
                    )
        ga_instance.run()
        # ga_instance.plot_fitness()