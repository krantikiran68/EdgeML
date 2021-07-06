# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

CC=g++
CFLAGS=-Wall -Wno-narrowing -p -g -fPIC -O3 -std=c++11 -pthread -ISoftPosit/source/include/ SoftPosit/build/Linux-x86_64-GCC/softposit.a

ifeq ($(DEBUG), True)
CFLAGS+= -DDEBUG
endif

PREDICTOR_INCLUDES=datatypes.h \
                   library_fixed.h \
                   library_float.h \
                   model_fixed.h \
                   model_float.h \
                   predictors.h \
                   profile.h \
                   vars_fixed.h \
                   vars_float.h \
                   library_posit.h \
                   vars_posit.h \
                   model_posit.h \

PREDICTOR_OBJS=debug.o \
               library_fixed.o \
               library_float.o \
               main.o \
               profile.o \
               seedot_fixed.o \
               seedot_float.o \
               library_posit.o \
               seedot_posit.o \

all: Predictor

Predictor: $(PREDICTOR_OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

debug.o: debug.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

library_fixed.o: library_fixed.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

library_float.o: library_float.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

main.o: main.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

profile.o: profile.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

seedot_fixed.o: seedot_fixed.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

seedot_float.o: seedot_float.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

library_posit.o: library_posit.cpp $(PREDICTOR_INCLUDES)
	$(CC) -c -o $@ $(CFLAGS) $<

clean: 
	rm -f *.o
	rm -f Predictor
