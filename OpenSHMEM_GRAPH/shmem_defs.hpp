#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>


#include <shmem.h>

int pWrk2[SHMEM_REDUCE_SYNC_SIZE];
long pSync[SHMEM_REDUCE_SYNC_SIZE];

typedef unsigned vertex_id_t;
typedef unsigned long long edge_id_t;

class Graph {
    public:
    /***
     The minimal graph repesentation consists of:
     n        -- the number of vertices
     m        -- the number of edges
     endV     -- an array of size m that stores the
                 destination ID of an edge <src->dest>.
     rowsIndices -- an array of size (n + 1) that stores pointers to the endV array (CRS format).
                 The degree of vertex i is given by rowsIndices[i + 1] - rowsIndices[i], and the
                 edges out of i are stored in the contiguous block
                 endV[rowsIndices[i] .. rowsIndices[i + 1] - 1].
     Vertices are numbered from 0 in our internal representation.
     ***/
    vertex_id_t n;
    edge_id_t m;
    edge_id_t *rowsIndices;
    vertex_id_t *endV;
    
    /* Distributed version variables */
    int nproc, rank;
    vertex_id_t local_n; /* local vertices number */
    vertex_id_t real_local_n;
    vertex_id_t local_first_vid; /* first vertex id */
    vertex_id_t local_last_vid; /* last vertex id */
    edge_id_t local_m; /* local edges number */
    vertex_id_t real_local_m;
    vertex_id_t endV_offset;

    void ReadGraph(const std::string& filename, int my_rank, int nproc);
    int GetPeByVertex(vertex_id_t v) const;
    int GetPeByEdge(edge_id_t e) const;
    int GetLocalVertexIndexByGlobal(vertex_id_t global_v) const;
};

int Graph::GetPeByVertex(vertex_id_t v) const {
    int rank = 0;
    while (rank * n / nproc <= v) ++rank;
    return rank - 1;
}

int Graph::GetPeByEdge(edge_id_t e) const {
    int rank = 0;
    while (rank * m / nproc <= e) ++rank;
    return rank - 1;
}

int Graph::GetLocalVertexIndexByGlobal(vertex_id_t global_v) const {
    int host_rank = GetPeByVertex(global_v);
    vertex_id_t left_boarder = host_rank * n / nproc;
    return global_v - left_boarder;
}

void Graph::ReadGraph(const std::string& filename, int my_rank, int nproc) {
    rank = my_rank;
    this->nproc = nproc;

    unsigned char align;
    std::ifstream in(filename);

    // Read global graph params
    in.read((char *)&n, sizeof(n));
    in.read((char *)&m, sizeof(m));
    in.read((char *)&align, sizeof(align));

    // Calculate vertex boarders
    vertex_id_t left_boarder = my_rank * n / nproc;
    vertex_id_t right_boarder = (my_rank + 1) * n / nproc;
    local_n = right_boarder - left_boarder;
    local_first_vid = left_boarder;
    local_last_vid = right_boarder - 1;
    int offset0 = sizeof(n) + sizeof(m) + sizeof(align);

    static int max_local_n, local_n_shmem = local_n;
    shmem_int_max_to_all(&max_local_n, &local_n_shmem, 1, 0, 0, nproc, pWrk2, pSync);
    real_local_n = max_local_n;

    // Read rowsIndices
    rowsIndices = (edge_id_t*)shmalloc(sizeof(edge_id_t) * (max_local_n + 1));
    int offset1 = offset0 + sizeof(edge_id_t) * left_boarder;
    in.seekg(offset1);
    in.read((char*)rowsIndices, sizeof(edge_id_t) * (local_n + 1));
    local_m = rowsIndices[local_n] - rowsIndices[0];

    static int max_local_m, local_m_shmem = local_m;
    shmem_int_max_to_all(&max_local_m, &local_m_shmem, 1, 0, 0, nproc, pWrk2, pSync);
    real_local_m = max_local_m;

    // Read edge ends
    endV = (vertex_id_t*)shmalloc(sizeof(vertex_id_t) * max_local_m);
    int offset2 = offset0 + sizeof(edge_id_t) * (n + 1) + sizeof(vertex_id_t) * rowsIndices[0];
    in.seekg(offset2);
    in.read((char*)endV, sizeof(vertex_id_t) * local_m);
    in.close();
}