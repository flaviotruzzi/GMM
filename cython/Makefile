schraudolph.o: schraudolph.c
	gcc schraudolph.c -c -o schraudolph.o
EMGMM.so: EMGMM.pyx schraudolph.o
	python2.7 setup.py build_ext --inplace --force
multivariateUtils.so: multivariateUtils.pyx
	python2.7 setup.py build_ext --inplace --force
