#include <fstream>
#include <string>
#include <map>
#include <iostream>
#include <vector>
#include <exception>
#include <ctime>

#include "mpi.h"

class RandomString {
	private:
	float p;
	std::string rule1;
	std::string rule2;
	public:
	RandomString() {}
	RandomString(float p, const std::string& rule1, const std::string rule2, int seed_offset = 0) {
		this->p = p;
		this->rule1 = rule1;
		this->rule2 = rule2;
		srand(time(NULL) + 123456 * seed_offset);
	}
	const std::string& GetString() const {
		if (float(rand()) / RAND_MAX > p) {
			return rule1;
		}
		return rule2;
	}

};

void SendResult(const std::string& data) {
	int size = data.size();
	MPI_Send(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	MPI_Send(data.c_str(), size + 1, MPI_SIGNED_CHAR, 0, 1, MPI_COMM_WORLD);
}

void DumpResult(const std::string& result, const std::string& save_prefix) {
	std::ofstream out(save_prefix + "_output.dat");
	out << result << std::endl;
	out.close();
}

void ReceiveAndDumpResult(const std::string& data, int ntasks, const std::string& save_prefix) {
	std::string result = data;
	MPI_Status status;
	for (int i = 1; i < ntasks; ++i) {
		int size;
		MPI_Recv(&size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
		char* buffer = new char[size + 1];
		MPI_Recv(buffer, size + 1, MPI_SIGNED_CHAR, i, 1, MPI_COMM_WORLD, &status);
		result += buffer;
		delete[] buffer;
	}
	DumpResult(result, save_prefix);
}

std::string UpdateData(const std::string& data, std::map<char, RandomString>& R) {
	std::string buf = "";
	for(int i = 0; i < data.length(); ++i) {
		buf += R[data[i]].GetString();
    }
	return buf;
}

void DumpLoad(int size, int t, int task_id, int ntasks, const std::string& save_prefix) {
	if (task_id) {
		MPI_Send(&size, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
		return;
	}
	std::vector<int> loads(ntasks, 0);
	loads[0] = size;

	MPI_Status status;
	int buffer;
	for (int i = 1; i < ntasks; ++i) {
		MPI_Recv(&buffer, 1, MPI_INT, i, 4, MPI_COMM_WORLD, &status);
		loads[i] = buffer;
	}

	std::ofstream out(save_prefix + "_stat.txt", std::ios_base::app);
	out << t << ",";
	for (int i = 0; i < loads.size(); ++i) {
		out << loads[i];
		if (i != loads.size() - 1) out << ",";
	}
	out << std::endl;
	out.close();
}

int GetCount(int size, int nei_size) {
	if (size > nei_size) {
		return (size - nei_size) / 2;
	}
	return 0;
}

std::string DirectedAlign(const std::string& data, int task_id, int ntasks, int prev_id, int next_id) {
	int size = data.size();
	int nei_size = size;

	MPI_Status status;

	MPI_Sendrecv(&size, 1, MPI_INT, prev_id, 0, &nei_size, 1, MPI_INT, next_id, 0, MPI_COMM_WORLD, &status);

	int c = GetCount(size, nei_size);
	int in_c = 0;
	std::string to_send;
	std::string new_data = data;
	if (data.size() >= c && !data.empty()) {
		to_send = data.substr(data.size() - c, c);
		new_data = data.substr(0, data.size() - c);
	}

	MPI_Sendrecv(&c, 1, MPI_INT, next_id, 1, &in_c, 1, MPI_INT, prev_id, 1, MPI_COMM_WORLD, &status);

	if (task_id % 2) {
		if (c) {
			MPI_Send(to_send.c_str(), c + 1, MPI_SIGNED_CHAR, next_id, 2, MPI_COMM_WORLD);
		}
		if (in_c) {
			char* buffer = new char[in_c + 1];
			MPI_Recv(buffer, in_c + 1, MPI_SIGNED_CHAR, prev_id, 2, MPI_COMM_WORLD, &status);
			new_data = buffer + new_data;
			delete[] buffer;
		}
	} else {
		if (in_c) {
			char* buffer = new char[in_c + 1];
			MPI_Recv(buffer, in_c + 1, MPI_SIGNED_CHAR, prev_id, 2, MPI_COMM_WORLD, &status);
			new_data = buffer + new_data;
			delete[] buffer;
		}
		if (c) {
			MPI_Send(to_send.c_str(), c + 1, MPI_SIGNED_CHAR, next_id, 2, MPI_COMM_WORLD);
		}
	}
	return new_data;
}

std::string AlignLoad(const std::string& data, int task_id, int ntasks) {
	int next_id = task_id + 1;
	int prev_id = task_id - 1;
	next_id = next_id < ntasks ? next_id : MPI_PROC_NULL;
	prev_id = prev_id >= 0 ? prev_id : MPI_PROC_NULL;

	auto new_data = DirectedAlign(data, task_id, ntasks, prev_id, next_id);
	new_data = DirectedAlign(new_data, task_id, ntasks, next_id, prev_id);
	return new_data;
}

void GetLsystemRules(const std::string& lsys_name, std::string& w0, std::map<char, RandomString>& R, int task_id) {
	if (lsys_name == "baba") {
		R['a'] = RandomString(-1, "b", "a", task_id);
		R['b'] = RandomString(-1, "ab", "b", task_id);
		w0 = "a";
		return;
	}
	if (lsys_name == "abbca") {
		R['a'] = RandomString(-1, "ab", "a", task_id);;
		R['b'] = RandomString(-1, "bc", "a", task_id);;
		w0 = "a";
		return;
	}
	if (lsys_name == "aaa") {
		R['a'] = RandomString(0.001, "a", "aa", task_id);
		w0 = "a";
		return;
	}
	if (lsys_name == "abaa") {
		R['a'] = RandomString(0.01, "a", "ab", task_id);
		R['b'] = RandomString(0.01, "b", "a", task_id);
		w0 = "a";
		return;
	}
	if (!task_id) {
		std::cout << "[ERROR]: " << "Incorrect lsys_name provided. Try one of the following instead:\n";
		std::cout << "* baba\n";
		std::cout << "* abbca\n";
		std::cout << "* aaa\n";
		std::cout << "* abaa\n";
		std::cout << "[OR] put your new L-system inside GetLsystemRules function.\n";
		throw std::invalid_argument("Unknown L-system rule");
	}
}

void RunLsystem(int T, int k, const std::string& lsys_name, int task_id, int ntasks, const std::string& save_prefix) {
	std::string data;
	std::map<char, RandomString> R;
	GetLsystemRules(lsys_name, data, R, task_id);
	if (task_id) {
		data.clear();
	}
	for(int t = 0; t < T; ++t) {
		if (!data.empty()) {
			data = UpdateData(data, R);
		}
		if (t % k == 0) {
			DumpLoad(data.size(), t, task_id, ntasks, save_prefix);
			data = AlignLoad(data, task_id, ntasks);
		}
    }
	if (!task_id) {
		ReceiveAndDumpResult(data, ntasks, save_prefix);
	} else {
		SendResult(data);
	}
}

void CleanUpFiles(const std::string& save_prefix, int ntasks) {
	std::ofstream out(save_prefix + "_stat.txt");
	out << "iteration";
	for (int i = 0; i < ntasks; ++i) {
		out << ",task_" << i;
	}
	out << std::endl;
	out.close();
	std::ofstream out2(save_prefix + "_output.dat");
	out2.close();
}

int main(int argc, char** argv) {
    int T = 20, k = 1;
	std::string lsys_name = "baba";
	std::string save_prefix = "";
    if (argc >= 4) {
	    T = std::atoi(argv[1]);
		k = std::atoi(argv[2]);
		lsys_name = argv[3];
    }

    int ntasks, task_id;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

	save_prefix = std::to_string(ntasks) + "_" + std::to_string(T) + "_" + std::to_string(k) + "_" + lsys_name;
	if (!task_id) {
		CleanUpFiles(save_prefix, ntasks);
	}

	RunLsystem(T, k, lsys_name, task_id, ntasks, save_prefix);

    MPI_Finalize();
	return 0;
}

