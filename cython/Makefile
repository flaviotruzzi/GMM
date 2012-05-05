EMGMM.so: EMGMM.pyx schraudolph.o
	python2.7 setup.py build_ext --inplace --force
schraudolph.o: schraudolph.c
	gcc schraudolph.c -c -o schraudolph.o -lm -DLITTLE_ENDIAN
