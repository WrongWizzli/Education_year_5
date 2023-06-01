#include <iostream>
#include <shmem.h>
#include <vector>
#include <queue>
#include <stack>
#include "defs_no_mpi.h"
#include "shmem_defs.hpp"
#include <string>
#include <chrono>
#include <unistd.h>

long long int pWrk[SHMEM_REDUCE_SYNC_SIZE];

volatile long lock = 0;
long long number_of_local_vertices;
using microseconds = uint64_t;

int BellmanFordSearch(const Graph& graph, int* dist, int* sigma) {
    int depth = 0;
    static long long int total_updates;
    static long long int local_updates;
    total_updates = 1;
    while(total_updates > 0) {
        local_updates = 0, total_updates = 0;
        for (vertex_id_t i = 0; i < graph.local_n; ++i) {
            if (dist[i] == depth) {
                for (edge_id_t edge = graph.rowsIndices[i]; edge < graph.rowsIndices[i + 1]; ++edge) {
                    vertex_id_t descendant = graph.endV[edge - graph.rowsIndices[0]];
                    int desc_owner = graph.GetPeByVertex(descendant);
                    vertex_id_t local_vid = graph.GetLocalVertexIndexByGlobal(descendant);
                    if (desc_owner == graph.rank) {
                        static int dist_stable;
                        dist_stable = dist[local_vid];
                        if (dist_stable == depth + 1 || dist_stable == -1) {
                            ++local_updates;
                            shmem_int_atomic_add(&sigma[local_vid], sigma[i], desc_owner);
                        }
                        if (dist_stable == -1) {
                            dist[local_vid] = depth + 1;
                        }
                    } else {
                        static int owner_dist, owner_sigma;
                        owner_dist = 0, owner_sigma = 0;
                        shmem_getmem_nbi(&owner_dist, dist + local_vid, sizeof(owner_dist), desc_owner);
                        if (owner_dist == depth + 1 || owner_dist == -1) {
                            ++local_updates;
                            shmem_int_atomic_add(&sigma[local_vid], sigma[i], desc_owner);
                        }
                        if (owner_dist == -1) {
                            owner_dist = depth + 1;
                            shmem_putmem_nbi(dist + local_vid, &owner_dist, sizeof(owner_dist), desc_owner);
                        }
                    }
                }
            }
        }
        shmem_longlong_sum_to_all(&total_updates, &local_updates, 1, 0, 0, graph.nproc, pWrk, pSync);
        ++depth;
    }
    // Synchronizing BFS before accumulation
    --depth;
    shmem_barrier_all();
    return depth;
}

void ActiveWaitLock(int desc_owner, int local_vid, int* semaphore) {
    int is_locked = 1;
    while (is_locked) {
        is_locked = shmem_int_atomic_compare_swap(&semaphore[local_vid], 0, 1, desc_owner);
        // Add nano_sleep and get semi-passive wait
    }
}

void ActiveWaitUnlock(int desc_owner, int local_vid, int* semaphore) {
    shmem_int_atomic_compare_swap(&semaphore[local_vid], 1, 0, desc_owner);
}

void DependencyAccumulation(const Graph& graph, int* dist, int* sigma, double* delta, double* delta_accumulative, double* bc, int bfs_depth, int* semaphore) {
    static double val;
    while (bfs_depth) {
        for (vertex_id_t i = 0; i < graph.local_n; ++i) {
            if (dist[i] == bfs_depth) bc[i] += delta[i];
        }
        shmem_barrier_all();
        for (vertex_id_t i = 0; i < graph.local_n; ++i) {
            if (dist[i] == bfs_depth) {
                for (edge_id_t edge = graph.rowsIndices[i]; edge < graph.rowsIndices[i + 1]; ++edge) {
                    vertex_id_t descendant = graph.endV[edge - graph.rowsIndices[0]];
                    int desc_owner = graph.GetPeByVertex(descendant);
                    vertex_id_t local_vid = graph.GetLocalVertexIndexByGlobal(descendant);
                    if (desc_owner == graph.rank) {
                        if (dist[local_vid] == bfs_depth - 1) {
                            val = (1. + delta[i]) * sigma[local_vid] / sigma[i];
                            ActiveWaitLock(desc_owner, local_vid, semaphore);
                            shmem_double_atomic_set(&delta_accumulative[local_vid], delta_accumulative[local_vid] + val, desc_owner);
                            ActiveWaitUnlock(desc_owner, local_vid, semaphore);
                        }
                    } else {
                        static int owner_dist, owner_sigma;
                        static double owner_delta;
                        shmem_getmem(&owner_dist, &dist[local_vid], sizeof(owner_dist), desc_owner);
                        if (owner_dist == bfs_depth - 1) {
                            shmem_getmem(&owner_sigma, &sigma[local_vid], sizeof(owner_sigma), desc_owner);
                            // If only I had atomic add double >_<
                            ActiveWaitLock(desc_owner, local_vid, semaphore);
                            shmem_double_atomic_set(
                                &delta_accumulative[local_vid], 
                                shmem_double_atomic_fetch(&delta_accumulative[local_vid], desc_owner) + double(1 + delta[i]) * owner_sigma / sigma[i],
                                desc_owner
                            );
                            ActiveWaitUnlock(desc_owner, local_vid, semaphore);
                        }
                    }
                }
            }
        }
        // Synchronize each reverse step
        shmem_barrier_all();
        for (int i = 0; i < graph.local_n; ++i) {
            delta[i] += delta_accumulative[i];
            delta_accumulative[i] = 0;
        }
        --bfs_depth;
        shmem_barrier_all();
    }
}

void ParseArgs(int argc, char** argv, std::string& in_file, std::string& out_file, bool& print_result) {
    const std::string in_flag{"-in"};
    const std::string out_flag{"-out"};
    const std::string should_print_result{"--show-result"};
    int i = 1;
    while (i < argc) {
        if (argv[i] == in_flag && i + 1 < argc) {
            in_file = argv[i + 1];
            ++i;
        }
        if (argv[i] == out_flag && i + 1 < argc) {
            out_file = argv[i + 1];
            ++i;
        }
        if (argv[i] == should_print_result) {
            print_result = true;
        }
        ++i;
    }
}

bool GetGraph(const std::string& in_file, graph_t* graph) {
    if (in_file.empty()) {
        std::cout << "Input file name cannot be missing. Skip exectuion...\n";
        return false;
    }
    char* in_file_char = new char[in_file.size()];
    for (size_t i = 0; i < in_file.size(); ++i) {
        in_file_char[i] = in_file[i];
    }
    readGraph(graph, in_file_char);
    delete[] in_file_char;
    return true;
}

void LogTimers(microseconds start_time,
               microseconds after_read,
               microseconds after_calculate,
               microseconds after_exchange) {
    std::cout << "Total time (mcs): " << after_exchange - start_time << std::endl;
    std::cout << "Read time: " << after_read - start_time << std::endl;
    std::cout << "Calculation time: " << after_calculate - after_read << std::endl;
    std::cout << "Exchange time: " << after_exchange - after_calculate << std::endl;
}

void LogProcDistribution(microseconds* timers, int nproc) {
    std::cout << "Processor time consume:\n";
    double sum = 0;
    for (int i = 0; i < nproc; ++i) {
        sum += timers[i];
    }
    for (int i = 0; i < nproc; ++i) {
        std::cout << std::round(timers[i] / sum * 10000) / 100. << "% ";
    }
    std::cout << std::endl;
}

microseconds GetTimeSinceEpochMcs() {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::microseconds>(current_time.time_since_epoch());
    return epoch.count();
}

void CleanBuffers(int* sigma, double* delta, double* delta_accumulative, int* dist, int* semaphore, int local_n) {
    for (int i = 0; i < local_n; ++i) {
        sigma[i] = 0;
        delta[i] = 0;
        delta_accumulative[i] = 0;
        semaphore[i] = 0;
        dist[i] = -1;
    }
}

void FreeShmemMemory(Graph* graph, double* delta, double* delta_accumulative, int* sigma, double* bc, int* dist, int* semaphore) {
    shmem_free(graph->endV);
    shmem_free(graph->rowsIndices);
    shmem_free(delta);
    shmem_free(delta_accumulative);
    shmem_free(sigma);
    shmem_free(bc);
    shmem_free(dist);
    shmem_free(semaphore);
}

void SummarizeResults(const Graph& graph, double* bc, bool print_result, const std::string& out_fname) {
    if (graph.rank == 0) {
        FILE *f = fopen(out_fname.c_str(), "wb");
        assert(f != NULL);
        if (print_result) std::cout << "Result:\n";
        for (int rank = 0; rank < graph.nproc; ++rank) {
            if (rank == graph.rank) {
                for (int i = 0; i < graph.local_n; ++i) {
                    if (print_result) std::cout << bc[i] << " ";
                    assert(fwrite(&bc[i], sizeof(double), 1, f) == 1);
                }
            } else {
                static long long int remote_number_of_vertices;
                static double bc_buffer;
                shmem_getmem(&remote_number_of_vertices, &number_of_local_vertices, sizeof(remote_number_of_vertices), rank);
                for (int i = 0; i < remote_number_of_vertices; ++i) {
                    shmem_getmem(&bc_buffer, &bc[i], sizeof(bc_buffer), rank);
                    if (print_result) std::cout << bc_buffer << " ";
                    assert(fwrite(&bc_buffer, sizeof(double), 1, f) == 1);
                }
            }
        }
        if (print_result) std::cout << std::endl << std::flush;
        fclose(f);
    }
}

int main(int argc, char** argv) {
    bool print_result{false};
    std::string in_file{};
    std::string out_file{"brandes_default_path"};
    ParseArgs(argc, argv, in_file, out_file, print_result);


    //OpenSHMEM inits
    int my_rank, nproc;
    shmem_init();
    my_rank = shmem_my_pe();
    nproc = shmem_n_pes();

    // Read Graph in SHMEMy way
    static auto start_time = GetTimeSinceEpochMcs();
    Graph graph;
    graph.ReadGraph(in_file, my_rank, nproc);
    shmem_barrier_all();
    static auto after_read = GetTimeSinceEpochMcs();
    // std::cout << "Rank: " << my_rank << " " << graph.local_n <<  " " << graph.local_m << " " << graph.local_last_vid << " " << graph.local_first_vid << std::endl;
    if (my_rank == 0) std::cout << "Total PEs: " << nproc << std::endl;
    shmem_barrier_all();

    // Buffers
    int* sigma = (int*) shmalloc(graph.real_local_n * sizeof(int));
    double* delta = (double*) shmalloc(graph.real_local_n * sizeof(double));
    double* delta_accumulative = (double*) shmalloc(graph.real_local_n * sizeof(double));
    double* bc = (double*) shmalloc(graph.real_local_n * sizeof(double));
    int* semaphore = (int*) shmalloc(graph.real_local_n * sizeof(int));
    int* dist = (int*) shmalloc(graph.real_local_n * sizeof(int));

    if (!my_rank) {
        std::cout << "True array size: " << graph.real_local_n << " " << graph.n << std::endl;
        std::cout << "True array size: " << graph.real_local_m << " " << graph.m << std::endl;
    }
    for (int i = 0; i < graph.real_local_n; ++i) {
        bc[i] = 0;
    }

    for (vertex_id_t v = 0; v < graph.n; ++v) {
        CleanBuffers(sigma, delta, delta_accumulative, dist, semaphore, graph.local_n);
        int vertex_rank = graph.GetPeByVertex(v);
        if (vertex_rank == graph.rank) {
            int local_vid = graph.GetLocalVertexIndexByGlobal(v);
            sigma[local_vid] = 1;
            dist[local_vid] = 0;
        }
        shmem_barrier_all();
        int bfs_depth = BellmanFordSearch(graph, dist, sigma);
        DependencyAccumulation(graph, dist, sigma, delta, delta_accumulative, bc, bfs_depth, semaphore);
    }
    for (int i = 0; i < graph.local_n; ++i) {
        bc[i] /= 2;
    }
    // Wait for all processes and make summary
    number_of_local_vertices = graph.local_n;
    shmem_barrier_all();

    SummarizeResults(graph, bc, print_result, out_file);

    // Free data and finish execution
    FreeShmemMemory(&graph, delta, delta_accumulative, sigma, bc, dist, semaphore);
    shmem_finalize();
    return 0;
}

// for (int i = 0; i < graph.nproc; ++i) {
//             if (my_pe() == i) {
//                 for (int j = 0; j < graph.local_n; ++j) {
//                     std::cout << dist[j] << " " << std::flush;
//                 }
//                 if(my_pe() == graph.nproc - 1) {
//                     std::cout << std::endl;
//                 }
//                 std::cout << std::flush;
//                 sleep(1);
//             }
//             shmem_barrier_all();
//         }
//         break;