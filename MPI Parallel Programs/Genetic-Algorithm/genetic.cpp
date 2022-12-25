#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <functional>
#include <iomanip>
#include <fstream>

#include "mpi.h"

std::string EXPERIMENT_ID;
std::string EXPERIMENT_ID_MEAN;
std::string EXPERIMENT_ID_BEST;

double frand() {
	return double(rand())/RAND_MAX;
}

double SphereFunction(const std::vector<double>& args) {
    double sum = 0;
    for (int i = 0; i < args.size(); ++i) {
        sum += args[i] * args[i];
    }
    return sum;
}

double RastriginFunction(const std::vector<double>& args) {
    double sum = 0;
    for (int i = 0; i < args.size(); ++i) {
        sum += args[i] * args[i] - 10 * cos(2 * M_PI * args[i]) + 10;
    }
    return sum;
}

double RozenbrokFunction(const std::vector<double>& args) {
    double sum = 0;
    for (int i = 0; i < args.size() - 1; ++i) {
        double s1 = (args[i] * args[i] - args[i + 1]);
        double s2 = (args[i] - 1);
        s1 = s1 * s1;
        s2 = s2 * s2;
        sum += 100 * s1 + s2;
    }
    return sum;
}

void InitVector(std::vector<std::vector<double>>& genes) {
	for(int i = 0; i < genes.size(); i++) {
		for(int j = 0; j < genes[i].size(); j++) {
			genes[i][j] = 100 - frand() * 200;
        }
    }
}

void Shuffle(std::vector<std::vector<double>>& genes) {
	for(int i = 0; i < genes.size(); ++i) {
		int l = rand() % genes.size();
		for(int j = 0; j < genes[i].size(); j++)
			std::swap(genes[i][j], genes[l][j]);
	}
}

void Select(std::vector<std::vector<double>>& genes, std::function<double(const std::vector<double>&)> f) {
	double win_proba = 0.85;
	Shuffle(genes);
	for(int i = 0; i < genes.size() / 2; i++) {
		int i0 = 2 * i;
		int i1 = 2 * i + 1;
		double sum_i0 = f(genes[i0]);
	    double sum_i1 = f(genes[i1]);
		double p = frand();
		if((sum_i0 < sum_i1 && p < win_proba) || (sum_i0 >= sum_i1 && p >= win_proba)) {
			for(int j = 0; j < genes[i0].size(); ++j) {
                genes[i1][j] = genes[i0][j];
            }
        } else {
			for(int j = 0; j < genes[i1].size(); ++j) {
				genes[i0][j] = genes[i1][j];
            }
        }
	}
}

void Crossover(std::vector<std::vector<double>>& genes) {
    Shuffle(genes);
	for(int i = 0; i < genes.size() / 2; ++i) {
		int i0 = 2 * i;
		int i1 = 2 * i + 1;
		int k = rand() % genes[i].size();
		for(int j = k; j < genes[i].size(); ++j) {
            std::swap(genes[i0][j], genes[i1][j]);
        }
	}
}

void Mutate(std::vector<std::vector<double>>& genes) {
	double pmut = 0.1;
	for(int i = 0; i < genes.size(); ++i) {
		for(int j = 0; j < genes[i].size(); ++j) {
			if (frand() < pmut) {
                genes[i][j] += (frand() - 0.5) / 2.5 * genes[i][j];
                genes[i][j] = std::max(genes[i][j], -100.0);
                genes[i][j] = std::min(genes[i][j], 100.0);
            }
        }
    }
}

void MigratePopulation(std::vector<std::vector<double>>& genes, int ntasks, int task_id) {
    int to_send_size = std::max(genes.size() * 0.1, 1.0);
    std::unordered_set<int> migration_indices_prev;
    std::unordered_set<int> migration_indices_next;
    while (migration_indices_prev.size() != to_send_size) {
        migration_indices_prev.emplace(rand() % genes.size());
    }
    while (migration_indices_next.size() != to_send_size) {
        migration_indices_next.emplace(rand() % genes.size());
    }

    auto p0_prev = migration_indices_prev.begin();
    auto p0_next = migration_indices_next.begin();
    int source_id_prev = task_id - 1;
    source_id_prev = source_id_prev < 0 ? ntasks - 1 : source_id_prev;
    int source_id_next = (task_id + 1) % ntasks;
    MPI_Status status;

    if (task_id % 2) {
        while (p0_next != migration_indices_next.end()) {
            int i = *p0_next;
            MPI_Send(&genes[i][0], genes[i].size(), MPI_DOUBLE, source_id_next, 0, MPI_COMM_WORLD);
            MPI_Recv(&genes[i][0], genes[i].size(), MPI_DOUBLE, source_id_next, 1, MPI_COMM_WORLD, &status);
            ++p0_next;
        }
        while (p0_prev != migration_indices_prev.end()) {
            int i = *p0_prev;
            MPI_Send(&genes[i][0], genes[i].size(), MPI_DOUBLE, source_id_prev, 0, MPI_COMM_WORLD);
            MPI_Recv(&genes[i][0], genes[i].size(), MPI_DOUBLE, source_id_prev, 1, MPI_COMM_WORLD, &status);
            ++p0_prev;
        }
    } else {
        std::vector<double> migrated_gene(genes[0].size());
        while (p0_prev != migration_indices_prev.end()) {
            int i = *p0_prev;
            MPI_Recv(&migrated_gene[0], genes[i].size(), MPI_DOUBLE, source_id_prev, 0, MPI_COMM_WORLD, &status);
            MPI_Send(&genes[i][0], genes[i].size(), MPI_DOUBLE, source_id_prev, 1, MPI_COMM_WORLD);
            std::copy(migrated_gene.begin(), migrated_gene.end(), genes[i].begin());
            ++p0_prev;
        }
        while (p0_next != migration_indices_next.end()) {
            int i = *p0_next;
            MPI_Recv(&migrated_gene[0], genes[i].size(), MPI_DOUBLE, source_id_next, 0, MPI_COMM_WORLD, &status);
            MPI_Send(&genes[i][0], genes[i].size(), MPI_DOUBLE, source_id_next, 1, MPI_COMM_WORLD);
            std::copy(migrated_gene.begin(), migrated_gene.end(), genes[i].begin());
            ++p0_next;
        }
    }
}

std::string ToString(const std::vector<double>& gene) {
    std::string s(gene.size(), ' ');
    for (int i = 0; i < gene.size(); ++i) {
        s[i] = gene[i] + '0';
    }
    return s;
}

int Argmin(const std::vector<double>& v) {
    int min_idx = 0;
    double min_val = v[0];
    for (int i = 1; i < v.size(); ++i) {
        if (v[i] < min_val) {
            min_val = v[i];
            min_idx = i;
        }
    }
    return min_idx;
}

void LogBestResults(const std::vector<double>& best_results, const std::vector<std::vector<double>>& best_genes,std::function<double(const std::vector<double>&)> f, int t) {
    if (t < 200) {
        std::cout << "Results: [ ";
        for (int i = 0; i < best_results.size(); ++i) {
            std::cout << best_results[i] << " ";
        }
        std::cout << "]" << std::endl;
    }
    int i_best = Argmin(best_results);
    if (t < 200) {
        std::cout << "Best setup: [ ";
        for (int j = 0; j < best_genes[i_best].size(); ++j) {
            std::cout << std::setprecision(3) << best_genes[i_best][j] << " ";
        }
        std::cout << "]" << std::endl;
    }
    auto best_result = f(best_genes[i_best]);
    double mean_result = 0.0;
    for (int i = 0; i < best_genes.size(); ++i) {
        mean_result += f(best_genes[i]) / best_genes.size();
    }
    if (t >= 200 && t % 1000 == 0) {
        std::cout << t << ": " << best_result << " / " << mean_result << std::endl;
    }
    std::ofstream out_best(EXPERIMENT_ID_BEST, std::ofstream::app);
    out_best << best_result << " ";
    out_best.close();
    std::ofstream out_mean(EXPERIMENT_ID_MEAN, std::ofstream::app);
    out_mean << mean_result << " ";
    out_mean.close();
}

void CollectBestResults(std::vector<std::vector<double>>& genes, int ntasks, int task_id, std::function<double(const std::vector<double>&)> f, int t) {
    int best_idx = 0;
	double best_sum = f(genes[0]);
	for(int i = 1; i < genes.size(); ++i) {
		double sum = f(genes[i]);
		if(sum < best_sum) {
			best_sum = sum;
			best_idx = i;
		}
	}
    if (!task_id) {
        MPI_Status status;
        std::vector<double> best_results(ntasks);
        std::vector<std::vector<double>> best_genes(ntasks, std::vector<double>(genes[0].size()));
        best_results[0] = best_sum;
        best_genes[0] = genes[best_idx];
        for (int i = 1; i < ntasks; ++i) {
            MPI_Recv(&best_results[i], 1, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&best_genes[i][0], best_genes[i].size(), MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
        }
        LogBestResults(best_results, best_genes, f, t);
    } else {
        MPI_Send(&best_sum, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&genes[best_idx][0], genes[best_idx].size(), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
}

void RunGeneticAlgorithm(int gene_size, int population_size, int n_steps, std::function<double(const std::vector<double>&)> f, int ntasks, int task_id) {
    std::vector<std::vector<double>> genes(population_size, std::vector<double>(gene_size, 0));
	InitVector(genes);
	 for(int t = 0; t < n_steps; t++) {
	 	Select(genes, f);
	 	Crossover(genes);
	 	Mutate(genes);
        MigratePopulation(genes, ntasks, task_id);
        CollectBestResults(genes, ntasks, task_id, f, t);
	 }
}

int main(int argc, char** argv) {
    int gene_size = -1, population_size = -1, n_steps = -1;
    std::string str_func("sphere");
    std::function<double(const std::vector<double>&)> f = SphereFunction;
    if (argc == 5) {
        gene_size = std::atoi(argv[1]);
        population_size = std::atoi(argv[2]);
        n_steps = std::atoi(argv[3]);
        str_func = std::string(argv[4]);
    }
    gene_size = gene_size > 0 ? gene_size : 10;
    population_size = population_size > 0 ? population_size : 20;
    n_steps = n_steps > 0 ? n_steps : 10;
    if (str_func == "rastrigin") {
        f = RastriginFunction;
    }
    if (str_func == "rozenbrok") {
        f = RozenbrokFunction;
    }
    int task_id, ntasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    EXPERIMENT_ID = str_func + "_" + std::to_string(gene_size) + "_" + std::to_string(population_size) + "_" + std::to_string(n_steps) + "_" + std::to_string(ntasks);
    EXPERIMENT_ID_BEST = EXPERIMENT_ID + "_best.txt";
    EXPERIMENT_ID_MEAN = EXPERIMENT_ID + "_mean.txt";
    if (!task_id) {
        std::cout << "N_tasks: " << ntasks << std::endl; 
    }
	RunGeneticAlgorithm(gene_size, population_size / ntasks, n_steps, f ,ntasks, task_id);

    MPI_Finalize();
	return 0;
}