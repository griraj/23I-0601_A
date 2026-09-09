CC ?= gcc
CFLAGS := -O3 -march=native -std=c11 -Wall -Wextra -Wpedantic -pthread -g -fno-omit-frame-pointer -Isrc
LDLIBS := -lopenblas -lm -pthread

TASK1 := task1_daxpy
TASK2 := task2_dgemv
TASK3 := task3_dgemm
TASK4 := task4_spmv
TARGETS := $(TASK1) $(TASK2) $(TASK3) $(TASK4)

all: $(TARGETS)

$(TASK1): src/$(TASK1).c src/common.h
	$(CC) $(CFLAGS) src/$(TASK1).c -o $@ $(LDLIBS)

$(TASK2): src/$(TASK2).c src/common.h
	$(CC) $(CFLAGS) src/$(TASK2).c -o $@ $(LDLIBS)

$(TASK3): src/$(TASK3).c src/common.h
	$(CC) $(CFLAGS) src/$(TASK3).c -o $@ $(LDLIBS)

# Task 4 has no BLAS dependency (no CBLAS sparse routine used), but we
# still link -lopenblas/-lm for consistency with the documented build command.
$(TASK4): src/$(TASK4).c src/common.h
	$(CC) $(CFLAGS) src/$(TASK4).c -o $@ $(LDLIBS)

run: all
	bash ./run_experiments.sh

clean:
	rm -f $(TARGETS) gmon.out

.PHONY: all run clean
