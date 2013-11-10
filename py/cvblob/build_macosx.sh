g++ -fPIC -I/usr/include/python2.7 -I/usr/include -I/usr/local/include -I/usr/local/include/opencv -c _cvblob.C
g++ -shared -Wl _cvblob.o -L/usr/lib -L/usr/local/lib -lopencv_core -lcvblob -lboost_python-mt -L/usr/lib/python2.7/config -lpython2.7 -o _cvblob.so
