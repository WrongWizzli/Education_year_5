# defines
MPICXX=mpicxx
SPEEDFLAGS=-O3
CXXFLAGS=-Wall $(SPEEDFLAGS)
LDFLAGS=$(SPEEDFLAGS)

TARGET = brandes_parallel_mashtab brandes_sequential gen_valid_info validation gen_RMAT gen_random

all: $(TARGET)

# RMAT generator
gen_RMAT: gen_RMAT.o graph_tools.o
	$(MPICXX) $^ -o $@ $(LDFLAGS)

# Erdos-Renyi (random) graph generator
gen_random: gen_random.o graph_tools.o
	$(MPICXX) $^ -o $@ $(LDFLAGS)

# gen_valid_info
gen_valid_info: gen_valid_info.o reference_bfs.o graph_tools.o
	$(MPICXX) $^ -o $@ $(LDFLAGS)

# gen_brandes_sequential
brandes_sequential: brandes.o graph_tools.o
	$(MPICXX) $^ -o $@ $(LDFLAGS)

# gen_brandes_sequential
brandes_parallel_mashtab: brandes_parallel_mashtab.o graph_tools_no_mpi.o
	oshc++ $^ -o $@ $(LDFLAGS)

# validation
validation: validation.o
	$(MPICXX) $^ -o $@ $(LDFLAGS)

.cpp.o:
	$(MPICXX) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -rf *.o $(TARGET)
