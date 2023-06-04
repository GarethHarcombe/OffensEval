make
time ./word2vec -train wiki-english-20171001 -output vectors.bin -cbow 0 -size 768 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15
./distance vectors.bin
