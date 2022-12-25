#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <assert.h>

void FillArrayRandomly(std::vector<int>& v, int srand_off) {
    srand(time(NULL) + srand_off);
    for (int i = 0; i < v.size(); ++i) {
        v[i] = rand() % 1000;
    }
}

void WriteArray(const std::vector<int>& v) {
    for (int i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

std::vector<int> JoinArrays(std::vector<int>& v1, std::vector<int>&v2) {
    std::vector<int> v;
    v.reserve(v1.size() + v2.size());
    v.insert(v.end(), v1.begin(), v1.end());
    v.insert(v.end(), v2.begin(), v2.end());
    return v;
}

void AddComparator(int first, int second, std::ofstream& out) {
    out << first << " " << second << std::endl;
}

void BuildScheduleRecursively(int first1, int first2, int step, int count1, int count2, std::ofstream& out) {
    if (count1 * count2 < 1) {
        return;
    }
    if (count1 == 1 && count2 == 1) {
        AddComparator(first1, first2, out);
        return;
    }

    int n1 = count1 / 2;
    int m1 = count2 / 2;
    BuildScheduleRecursively(first1, first2, 2 * step, count1 - n1, count2 - m1, out);
    BuildScheduleRecursively(first1 + step, first2 + step, 2 * step, n1, m1, out);

    int i = 1;
    for (; i < count1 - 1; i += 2) {
        AddComparator(first1 + step * i, first1 + step * (i + 1), out);
    }

    if (count1 % 2 == 0) {
        AddComparator(first1 + step * (count1 - 1), first2, out);
        i = 1;
    } else {
        i = 0;
    }
    for (; i < count2 - 1; i += 2) {
        AddComparator(first2 + step * i, first2 + step * (i + 1), out);
    }
}

void BuildSchedule(int p1, int p2) {
    std::ofstream out("comparator.txt");
    BuildScheduleRecursively(0, p1, 1, p1, p2, out);
    out.close();
}

void SortJoinedArrayByComps(std::vector<int>& v, std::ifstream& in) {
    std::vector<int> v_copy = v;
    std::sort(v_copy.begin(), v_copy.end());

    int i0, i1, k = 0;
    while (in >> i0 >> i1) {
        std::cout << i0 << " " << i1 << std::endl;
        if (v[i0] > v[i1]) {
            std::swap(v[i0], v[i1]);
        }
        k++;
    }
    std::cout << k << std::endl;
    assert(v == v_copy);
}

int CountTacts(int size) {
    std::ifstream in("comparator.txt");
    std::vector<int> max_tacts(size, 0);
    int i0, i1;
    while (in >> i0 >> i1) {
        max_tacts[i0] = std::max(max_tacts[i0], max_tacts[i1]) + 1;
        max_tacts[i1] = max_tacts[i0];
    }
    return *std::max_element(max_tacts.begin(), max_tacts.end());
}

void ValidateSchedule(std::vector<int>& v1, std::vector<int>& v2) {
    std::ifstream in("comparator.txt");
    std::vector<int> v = JoinArrays(v1, v2);

    std::cout << v1.size() << " " << v2.size() << " " << 0 << std::endl;

    SortJoinedArrayByComps(v, in);

    std::cout << CountTacts(v.size()) << std::endl;
}

void TryFirst24() {
    for (int sum_len = 1; sum_len < 25; ++sum_len) {
        for (int len_first = 1; len_first <= sum_len; ++len_first) {
            BuildSchedule(len_first, sum_len - len_first);
            for (int n_zeros1 = 0; n_zeros1 <= len_first; ++n_zeros1) {
                std::vector<int> v1(len_first, 1);
                for (int i = 0; i < n_zeros1; ++i) {
                    v1[i] = 0;
                }
                for (int n_zeros2 = 0; n_zeros2 <= sum_len - len_first; ++n_zeros2) {
                    std::vector<int> v2(sum_len - len_first, 1);
                    for (int i = 0; i < n_zeros2; ++i) {
                        v2[i] = 0;
                    }
                    ValidateSchedule(v1, v2);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "p1 or p2 not provided as CMD arguements. Please, type them!\n";
        return 1;
    }
    int p1 = std::atoi(argv[1]);
    int p2 = std::atoi(argv[2]);

    // TryFirst24();

    std::vector<int> v1(p1), v2(p2);
    FillArrayRandomly(v1, 1);
    FillArrayRandomly(v2, 2);
    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    // WriteArray(v1);
    // WriteArray(v2);

    BuildSchedule(p1, p2);
    ValidateSchedule(v1, v2);

    return 0;
}