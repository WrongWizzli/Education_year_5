#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <string>

#include "mpi.h"

std::random_device rand_dev;
std::mt19937 generator(rand_dev());
std::uniform_real_distribution<double> distr(0, 1);

void dump_experiment_info(std::vector<int64_t>& bouts, std::vector<int64_t>& ts, std::vector<int64_t>& ns, const std::string& save_prefix) {
    std::ofstream out(save_prefix + "_out.txt");
    int64_t total_bout = 0;
    int64_t total_t = 0;
    int64_t total_n = 0;
    for (int i = 0; i < bouts.size(); ++i) {
        total_bout += bouts[i];
        total_t += ts[i];
        total_n += ns[i];
    }
    out << "{\"p\": " << total_bout / double(total_n);
    out << ", \"t\": " << total_t / double(total_n) << "}";
}

void dump_exec_info(double sendrecv_time, double max_loop_time, int ntasks, const std::string& save_prefix) {
    std::ofstream out(save_prefix + "_stat.txt");
    out << "{\"Tsendrecv\": " << sendrecv_time;
    out << ", \"Tmaxloop\": " << max_loop_time;
    out << ", \"Ntasks\": " << ntasks << "}";
}

int64_t do_walk(int a, int b, int x, double p, int64_t& t) {
    while (x > a && x < b) {
        if (distr(generator) < p) {
            x++;
        } else {
            x--;
        }
        t++;
    }
    return x;
}

void do_walks(int a, int b, int x, double p, int n, int task_id, int ntasks, const std::string& save_prefix) {
    int64_t t = 0;
    int64_t bout = 0;
    auto t0 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (int i = 0; i < n; ++i) {
        int out_walk = do_walk(a, b, x, p, t);
        if (out_walk == b) {
            bout++;
        }
    }
    auto t1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    double loop_time = t1 - t0;
    if (!task_id) {
        std::vector<int64_t> bouts(ntasks), ts(ntasks), ns(ntasks);
        std::vector<double> loop_times(ntasks);
        bouts[0] = bout;
        ts[0] = t;
        ns[0] = n;
        loop_times[0] = loop_time;
        for (int i = 1; i < ntasks; ++i) {
            MPI_Status status;
            int64_t rbout, rt;
            int rn;
            double loop_time;
            MPI_Recv(&rbout, 1, MPI_LONG_LONG_INT, i, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&rt, 1, MPI_LONG_LONG_INT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&rn, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&loop_time, 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, &status);
            auto source_id = status.MPI_SOURCE;
            bouts[source_id] = rbout;
            ts[source_id] = rt;
            ns[source_id] = rn;
            loop_times[source_id] = loop_time;
        }
        auto t2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        dump_experiment_info(bouts, ts, ns, save_prefix);
        dump_exec_info(
            t2 - t1, 
            *std::max_element(loop_times.cbegin(), loop_times.cend()),
            ntasks,
            save_prefix
        );
    }  else {
        MPI_Send(&bout, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&t, 1, MPI_LONG_LONG_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&n, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&loop_time, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {

    int task_id, ntasks;
    int a = std::atoi(argv[1]);
    int b = std::atoi(argv[2]);
    int x = std::atoi(argv[3]);
    double p = std::atof(argv[4]);
    int n = std::atoi(argv[5]);
    std::string save_prefix = argv[6];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    do_walks(a, b, x, p, n / ntasks, task_id, ntasks, save_prefix);
    
    MPI_Finalize();
    return 0;
}