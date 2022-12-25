#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

int64_t GetEpochMicroSeconds() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void DumpArrayToFile(int** data, int field_size, const std::string& prefix = "_") {
    std::ofstream out(prefix + "seq_output.dat");
	for(int i = 0; i < field_size; ++i) {
		for(int j = 0; j < field_size; ++j) {
			out << data[i][j] << " ";
        }
		out << std::endl;
	}
	out.close();
}

void DumpMetaDataToFile(int n_iter, int field_size, int first_repeat, int64_t loop_time) {
    std::ofstream out("seq_stat.txt");
    out << "N iterations: " << n_iter << std::endl;
    out << "Field size: " << field_size << std::endl;
    out << "First iteration of start layout repeatance: " << first_repeat << std::endl;
    out << "Loop time (mcs): " << loop_time << std::endl;
}

int CalculateNewState(int** data, int i, int j, int field_size) {
	int state = data[i][j];
	int s = -state;
	for(int ii = i - 1; ii <= i + 1; ++ii) {
		for(int jj = j - 1; jj <= j + 1; ++jj) {
            int ii_bounded = ii < 0 ? field_size - 1 : ii;
            ii_bounded = ii_bounded == field_size ? 0 : ii_bounded;
            int jj_bounded = jj < 0 ? field_size - 1 : jj;
            jj_bounded = jj_bounded == field_size ? 0 : jj_bounded;
            s += data[ii_bounded][jj_bounded];
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

void UpdateData(int field_size, int** data, int** temp) {
	for(int i = 0; i < field_size; ++i) {
		for(int j = 0; j < field_size; ++j) {
			temp[i][j] = CalculateNewState(data, i, j, field_size);
        }
    }
}

void SetupFromFile(int field_size, int** data, int** data_0) {
    std::ifstream in("start_config.txt");
    int i, j;
    while (in >> i >> j) {
        if (i >= field_size || j >= field_size) {
            std::cout << "WARN! Input configuration is out of field: ";
            std::cout << "[ " << i << ", " << j << " ].";
            std::cout << "Max expected index for (i, j): " << field_size - 1 << std::endl;
            continue;
        }
        data_0[i][j] = data[i][j] = 1;
    }
}

bool IsRepeatingStart(int** data, int** data_0, int field_size) {
    for (int i = 0; i < field_size; ++i) {
        for (int j = 0; j < field_size; ++j) {
            if (data[i][j] != data_0[i][j]) {
                return false;
            }
        }
    }
    return true;
}

void InitField(int field_size, int** data, int** data_0, int** temp) {
	for (int i = 0; i < field_size; ++i) {
        for (int j = 0; j < field_size; ++j) {
            data_0[i][j] = data[i][j] = temp[i][j] = 0;
        }
    }
	SetupFromFile(field_size, data, data_0);
}

void RunLife(int field_size, int n_iter) {
	int** data = new int*[field_size];
    for (int i = 0; i < field_size; ++i) {
        data[i] = new int[field_size];
    }
	int** temp = new int*[field_size];
    for (int i = 0; i < field_size; ++i) {
        temp[i] = new int[field_size];
    }
    int** data_0 = new int*[field_size];
    for (int i = 0; i < field_size; ++i) {
        data_0[i] = new int[field_size];
    }
	InitField(field_size, data, data_0, temp);
    DumpArrayToFile(data_0, field_size, "start_state_");

    int64_t loop_time = 0;
    int first_repeat = -1;
	for (int t = 0; t < n_iter; ++t) {
        loop_time -= GetEpochMicroSeconds();
		UpdateData(field_size, data, temp);
		std::swap(data, temp);
        loop_time += GetEpochMicroSeconds();
        if (first_repeat == -1 && IsRepeatingStart(data, data_0, field_size)) {
            first_repeat = t + 1;
        }
	}
    DumpArrayToFile(data, field_size, "final_state_");
    DumpMetaDataToFile(n_iter, field_size, first_repeat, loop_time);

	for (int i = 0; i < field_size; ++i) {
        delete[] data[i];
        delete[] data_0[i];
        delete[] temp[i];
    }
	delete[] data;
    delete[] data_0;
    delete[] temp;
}

int main(int argc, char** argv) {
    int field_size = 15, n_iter = 20;
    if (argc >= 3) {
	    field_size = std::atoi(argv[1]);
	    n_iter = std::atoi(argv[2]);
    }
	RunLife(field_size, n_iter);
	return 0;
}
