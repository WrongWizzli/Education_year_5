#include <fstream>
#include <iostream>
#include <string>
#include <string>
#include <chrono>
#include <cmath>

#include "mpi.h"

int CalculateNewState(int* data, int i, int j, int field_size) {
	int state = data[i * (field_size + 2) + j];
	int s = -state;
	for(int ii = i - 1; ii <= i + 1; ++ii) {
		for(int jj = j - 1; jj <= j + 1; ++jj) {
            s += data[ii * (field_size + 2) + jj];
        }
    }
	if(state == 0 && s == 3) {
		return 1;
    }
	if(state == 1 && (s < 2 || s > 3)) {
		return 0;
    }
	return state;
}

void UpdateData(int field_size, int* data, int* temp) {
	for (int i = 1; i < field_size + 1; ++i) {
        for (int j = 1; j < field_size + 1; ++j) {
			temp[i * (field_size + 2) + j] = CalculateNewState(data, i, j, field_size);
        }
    }
}

void RecvDataWorker(int* temp, int n_workers, int sender_id) {
    MPI_Status status;
    MPI_Recv(temp, n_workers, MPI_INT, sender_id, 0, MPI_COMM_WORLD, &status);
}

void SendDataWorker(int* temp, int n_workers, int recv_id) {
    MPI_Send(temp, n_workers, MPI_INT, recv_id, 1, MPI_COMM_WORLD);
}

void RunLifeWorker(int field_size, int n_iter, int task_id, int ntasks) {
    int cell_size = (field_size + 2) * (field_size + 2);
	int* data = new int[cell_size];
    int* temp = new int[cell_size];
	for (int t = 0; t < n_iter; ++t) {
        RecvDataWorker(data, cell_size, ntasks - 1);
        UpdateData(field_size, data, temp);
        std::swap(data, temp);
        SendDataWorker(data, cell_size, ntasks - 1);
	}
    delete[] temp;
    delete[] data;
}

int64_t GetEpochMicroSeconds() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void DumpArrayToFile(int** data, int field_size, int n_fields, const std::string& prefix = "_") {
    int n_rows = std::sqrt(n_fields);
    int full_size = n_rows * field_size;
    int **tmp = new int*[full_size];
    for (int i = 0; i < full_size; ++i) {
        tmp[i] = new int[full_size];
    }
    for (int i = 0; i < full_size; ++i) {
        for (int j = 0; j < full_size; ++j) {
            int row_id = i / field_size;
            int col_id = j / field_size;
            tmp[i][j] = data[row_id * n_rows + col_id][(i % field_size + 1) * (field_size + 2) + j % field_size + 1];
        }
    }
    std::ofstream out(prefix + "output.dat");
	for(int i = 0; i < full_size; ++i) {
		for(int j = 0; j < full_size; ++j) {
			out << tmp[i][j] << " ";
        }
		out << std::endl;
	}
	out.close();
    for (int i = 0; i < full_size; ++i) {
        delete[] tmp[i];
    }
    delete[] tmp;
}

void DumpMetaDataToFile(int n_iter, int field_size, int first_repeat, int64_t loop_time, const std::string& prefix_path) {
    std::ofstream out(prefix_path + "stat.txt");
    out << "N iterations: " << n_iter << std::endl;
    out << "Field size: " << field_size << std::endl;
    out << "First iteration of start layout repeatance: " << first_repeat << std::endl;
    out << "Loop time (mcs): " << loop_time << std::endl;
}

bool IsRepeatingStart(int** data, int** data_0, int n_workers, int field_size, const std::string& prefix_path) {
    for (int i = 0; i < n_workers; ++i) {
        for (int j = 1; j < field_size + 1; ++j) {
            for (int k = 1; k < field_size + 1; ++k) {
                if (data[i][j * (field_size + 2) + k] != data_0[i][j * (field_size + 2) + k]) {
                    return false;
                }
            }
        }
    }
    DumpArrayToFile(data, field_size, n_workers, prefix_path + "cur_state_");
    return true;
}

void SetupFromFile(int** data, int** data_0, int field_size, int n_workers) {
    std::ifstream in("start_config.txt");
    int i, j, k;
    while (in >> i >> j >> k) {
        if (i >= n_workers || j >= field_size) {
            std::cout << "WARN! Input configuration is out of field: ";
            std::cout << "[ " << i << ", " << j << ", " << k << " ].";
            std::cout << "Max expected index for (i, j, k): (" << n_workers - 1 << ", " << field_size - 1 << ", " << field_size - 1 << ")" << std::endl;
            continue;
        }
        int inner_idx = (j + 1) * (field_size + 2) + (k + 1);
        data[i][inner_idx] = data_0[i][inner_idx] = 1;
    }
}

void InitField(int** data, int** data_0, int field_size, int n_workers) {
    int cell_size = (field_size + 2) * (field_size + 2);
	for (int i = 0; i < n_workers; ++i) {
        for (int j = 0; j < cell_size; ++j) {
            data_0[i][j] = data[i][j] = 0;
        }
    }
	SetupFromFile(data, data_0, field_size, n_workers);
}

int SyncBoundary(int** data, int i, int j, int k, int cell_shape, int n_workers) {
    int n_rows = std::sqrt(n_workers);
    int i_row = i / n_rows;
    int j_row = i % n_rows;
    int r_j = j, r_k = k;
    if (j == 0) {--i_row; r_j = cell_shape - 2;}
    if (k == 0) {--j_row; r_k = cell_shape - 2;}
    if (j == cell_shape - 1) {++i_row; r_j = 1;}
    if (k == cell_shape - 1) {++j_row; r_k = 1;}
    i_row = i_row < 0 ? n_rows - 1 : i_row;
    i_row = i_row == n_rows ? 0 : i_row;
    j_row = j_row < 0 ? n_rows - 1 : j_row;
    j_row = j_row == n_rows ? 0 : j_row;
    return data[i_row * n_rows + j_row][r_j * cell_shape + r_k];
}

void SetupBoundaries(int** data, int field_size, int n_workers) {
    int cell_shape = field_size + 2;
    for (int i = 0; i < n_workers; ++i) {
        for (int j = 0; j < cell_shape; ++j) {
            for (int k = 0; k < cell_shape; ++k) {
                if (j == 0 || k == 0 || j == cell_shape - 1 || k == cell_shape - 1) {
                    data[i][j * cell_shape + k] = SyncBoundary(data, i, j, k, cell_shape, n_workers);
                }
            }
        }
    }
}

void RunLifeMain(int field_size, int n_iter, int task_id, int ntasks, const std::string& prefix_path) {
    int cell_size = (field_size + 2) * (field_size + 2);

    int** data = new int*[ntasks - 1];
    for (int i = 0; i < ntasks - 1; ++i) {
        data[i] = new int[cell_size];
    }
    int** data_0 = new int*[ntasks - 1];
    for (int i = 0; i < ntasks - 1; ++i) {
        data_0[i] = new int[cell_size];
    }
    InitField(data, data_0, field_size, ntasks - 1);
    DumpArrayToFile(data_0, field_size, ntasks - 1, prefix_path + "start_state_");

    int64_t loop_time = 0;
    int first_repeat = -1;
    int *cell = new int[cell_size];
    MPI_Status status;
    for (int t = 0; t < n_iter; ++t) {
        loop_time -= GetEpochMicroSeconds();
        SetupBoundaries(data, field_size, ntasks - 1);
        for (int i = 0; i < ntasks - 1; ++i) {
            MPI_Send(data[i], cell_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        for (int i = 0; i < ntasks - 1; ++i) {
            MPI_Recv(cell, cell_size, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            std::swap(cell, data[status.MPI_SOURCE]);
        }
        loop_time += GetEpochMicroSeconds();
        if (first_repeat == -1 && IsRepeatingStart(data, data_0, ntasks - 1, field_size, prefix_path)) {
            first_repeat = t + 1;
        }
	}
    DumpArrayToFile(data, field_size, ntasks - 1, prefix_path + "final_state_");
    DumpMetaDataToFile(n_iter, field_size, first_repeat, loop_time, prefix_path);
    for (int i = 0; i < ntasks - 1; ++i) {
        delete[] data[i];
        delete[] data_0[i];
    }
    delete[] data;
    delete[] data_0;
}

int ObtainIntSqrt(int ntasks) {
    double sq_task = std::sqrt(ntasks - 1);
    while (ntasks >= 1 && sq_task != int(sq_task)) {
        --ntasks;
        sq_task = std::sqrt(ntasks - 1);
    }
    return ntasks;
}

std::string GetLogPrefixPath(int ntasks, int field_size, int n_iter) {
    std::string prefix("runs/");
    prefix += std::to_string(ntasks);
    prefix += "_";
    prefix += std::to_string(field_size);
    prefix += "_";
    prefix += std::to_string(n_iter);
    prefix += "_";
    return prefix;
}

int main(int argc, char** argv) {
    int field_size = 10, n_iter = 1;
    if (argc >= 3) {
	    field_size = std::atoi(argv[1]);
	    n_iter = std::atoi(argv[2]);
    }
    int task_id, ntasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    ntasks = ObtainIntSqrt(ntasks);
    const auto& prefix_path = GetLogPrefixPath(ntasks, field_size, n_iter);

    if (ntasks >= 2 && task_id < ntasks) {
        if (task_id == ntasks - 1) {
            RunLifeMain(field_size, n_iter, task_id, ntasks, prefix_path);
        } else {
	        RunLifeWorker(field_size, n_iter, task_id, ntasks);
        }
    }
    MPI_Finalize();
	return 0;
}
