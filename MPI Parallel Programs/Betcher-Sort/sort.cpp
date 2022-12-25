#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <exception>
#include <stdexcept>
#include <ios>

#include "mpi.h"

#define MERGE_STEP 1

char SORT_MODE = 'x';
int TACT_COUNT = 0;

struct Point {
    float x;
    float y;
    Point() {}
    Point(bool need_randomize) {
        x = rand() % 10000;
        y = rand() % 10000;
    }
    bool operator<(const Point& c) const {
        if (SORT_MODE == 'x') {
            return x < c.x;
        }
        return y < c.y;
    }
    bool operator>(const Point& c) const {
        if (SORT_MODE == 'x') {
            return x > c.x;
        }
        return y > c.y;
    }
    bool operator>=(const Point& c) const {
        if (SORT_MODE == 'x') {
            return x >= c.x;
        }
        return y >= c.y;
    }
    bool operator<=(const Point& c) const {
        if (SORT_MODE == 'x') {
            return x <= c.x;
        }
        return y <= c.y;
    }
};

void SubMergeArrays(const std::vector<Point>& points_i, const std::vector<Point>& points_j, std::vector<Point>& all_points) {
    int i = 0;
    int j = 0;
    int k = 0;

    while (i < points_i.size() && j < points_j.size()) {
        if (points_i[i] < points_j[j]) {
            all_points[k++] = points_i[i++];
        } else {
            all_points[k++] = points_j[j++];
        }
    }
    while (i < points_i.size()) {
        all_points[k++] = points_i[i++];
    }
    while (j < points_j.size()) {
        all_points[k++] = points_j[j++];
    }
}


void SubMerge(int task_id, int p_i, int p_j, std::vector<Point> &points_i) {
    if (task_id != p_i && task_id != p_j) return;
    int cop_tacts;
    if (task_id == p_j) {
        MPI_Send(points_i.data(), points_i.size() * sizeof(Point), MPI_BYTE, p_i, 0, MPI_COMM_WORLD);
        MPI_Send(&TACT_COUNT, 1, MPI_INT, p_i, 0, MPI_COMM_WORLD);
        MPI_Recv(points_i.data(), points_i.size() * sizeof(Point), MPI_BYTE, p_i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&cop_tacts, 1, MPI_INT, p_i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        TACT_COUNT = std::max(TACT_COUNT, cop_tacts) + 1;
        return;
    }

    std::vector<Point> points_j(points_i.size());
    MPI_Recv(points_j.data(), points_i.size() * sizeof(Point), MPI_BYTE, p_j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&cop_tacts, 1, MPI_INT, p_j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<Point> all_points(2 * points_i.size());
    SubMergeArrays(points_i, points_j, all_points);
    MPI_Send(all_points.data() + points_i.size(), points_i.size() * sizeof(Point), MPI_BYTE, p_j, 0, MPI_COMM_WORLD);
    MPI_Send(&TACT_COUNT, 1, MPI_INT, p_j, 0, MPI_COMM_WORLD);
    TACT_COUNT = std::max(TACT_COUNT, cop_tacts) + 1;

    for (int i = 0; i < points_i.size(); ++i) {
        points_i[i] = all_points[i];
    }
}

void MergeRecursively(int task_id, int first1, int first2, int step, int count1, int count2, std::vector<Point>& points) {
    if (count1 * count2 < 1) {
        return;
    }
    if (count1 == 1 && count2 == 1) {
        SubMerge(task_id, first1, first2, points);
        return;
    }

    int n1 = count1 / 2;
    int m1 = count2 / 2;

    MergeRecursively(task_id, first1, first2, 2 * step, count1 - n1, count2 - m1, points);
    MergeRecursively(task_id, first1 + step, first2 + step, 2 * step, n1, m1, points);

    int i = 1;
    for (; i < count1 - 1; i += 2) {
        SubMerge(task_id, first1 + step * i, first1 + step * (i + 1), points);
    }

    if (count1 % 2 == 0) {
        SubMerge(task_id, first1 + step * (count1 - 1), first2, points);
        i = 1;
    } else {
        i = 0;
    }
    for (; i < count2 - 1; i += 2) {
        SubMerge(task_id, first2 + step * i, first2 + step * (i + 1), points);
    }
}

void BetcherSortRecursively(int task_id, int pos0, int ntasks, std::vector<Point>& points) {
    if (ntasks <= 1) { 
        return; 
    }
    int mid = ntasks / 2;
    BetcherSortRecursively(task_id, pos0, mid, points);
    BetcherSortRecursively(task_id, pos0 + mid, ntasks - mid, points);
    MergeRecursively(task_id, pos0, pos0 + mid, MERGE_STEP, mid, ntasks - mid, points);
}

std::string GenerateSavePathPrefix(int ntasks, char sort_mode, int x_size, int y_size) {
    std::stringstream ss;
    ss << ntasks << "_";
    ss << SORT_MODE << "_";
    ss << x_size << "_";
    ss << y_size;
    std::string save_path;
    ss >> save_path;
    return save_path;
}

bool ValidateResult(const std::vector<Point>& points, int ntasks) {
    bool is_broken = false;
    std::vector<Point> result(ntasks * points.size());
    for (int i = 0; i < points.size(); ++i) {
        result[i] = points[i];
    }
    std::vector<Point> buffer(points.size());
    for (int i = 1; i < ntasks; ++i) {
        MPI_Recv(buffer.data(), buffer.size() * sizeof(Point), MPI_BYTE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < buffer.size(); ++j) {
            result[i * points.size() + j] = buffer[j];
        }
    }
    std::cout << "SORT_MODE: " << SORT_MODE << std::endl;
    for (int i = 0; i < result.size() - 1; ++i) {
        if (result[i] > result[i + 1]) {
            is_broken = true;
            std::cout << "[ERROR]. Element with idx [" << i << "] is bigger than right neighbour. ";
            std::cout << "(" << result[i].x << ", " << result[i].y << ") vs (" << result[i + 1].x << ", " << result[i + 1].y << ")\n";
        }
    }
    std::cout << "======================\n";
    if (result.size() < 500) {
        for (int i = 0; i < result.size(); ++i) {
            std::cout << "(" << result[i].x << ", " << result[i].y << ")" << std::endl;
        }
    }
    return is_broken;
}

void DumpResultToFile(int x_size, int y_size, int ntasks, char sort_mode, double t_max, bool is_broken, int tact_number, const std::string& save_path_prefix) {
    std::ofstream out((save_path_prefix + "_result.txt").c_str());
    out << "{\n \"x_size\": " << x_size << ",\n";
    out << " \"y_size\": " << y_size << ",\n";
    out << " \"ntasks\": " << ntasks << ",\n";
    out << " \"sort_mode\": " << sort_mode << ",\n";
    out << " \"t_max\": " << t_max << ",\n";
    out << " \"is_broken\": " << is_broken << "\n";
    out << " \"tact_number\": " << tact_number << "\n";
    out << "}";
    out.close();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        throw std::invalid_argument("Too few arguments in command line. Expected 2 at least (grid size).");
    }
    int x_size = std::atoi(argv[1]);
    int y_size = std::atoi(argv[2]);
    if (argc >= 4) SORT_MODE = argv[3][0];
    int task_id, ntasks;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    srand(ntasks);
    const std::string& save_path_prefix = GenerateSavePathPrefix(ntasks, SORT_MODE, x_size, y_size);
    int size = x_size * y_size;
    int size_points = (size + ntasks - 1) / ntasks;
    std::vector<Point> points(size_points);
    for (int i = 0; i < points.size(); ++i) {
        points[i] = Point(true);
        if (i >= size) {
            points[i].x = 100.0;
            points[i].y = 100.0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();
    std::sort(points.begin(), points.end());
    BetcherSortRecursively(task_id, 0, ntasks, points);
    double t = MPI_Wtime();

    double dt = t - t0;
    double t_max;
    double tact_max;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&dt, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&TACT_COUNT, &tact_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!task_id) {
        bool is_broken = ValidateResult(points, ntasks);
        DumpResultToFile(x_size, y_size, ntasks, SORT_MODE, t_max, is_broken, TACT_COUNT, save_path_prefix);
    } else {
        MPI_Send(points.data(), points.size() * sizeof(Point), MPI_BYTE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}





