for i in 64 32 16 8 4 2
do

echo "=== threads:" ${i} "==="

echo "./cpp-data/small-lsvm-0"
./estimator ./cpp-data/small-lsvm-0 0.03125 10 ${i}
echo "./cpp-data/small-lsvm-1"
./estimator ./cpp-data/small-lsvm-1 0.1 10 ${i}
echo "./cpp-data/small-lsvm-2"
./estimator ./cpp-data/small-lsvm-2 0.1 10 ${i}
echo "./cpp-data/small-lsvm-3"
./estimator ./cpp-data/small-lsvm-3 0.1 10 ${i}
echo "./cpp-data/small-lsvm-4"
./estimator ./cpp-data/small-lsvm-4 0.1 10 ${i}
echo "./cpp-data/small-lsvm-5"
./estimator ./cpp-data/small-lsvm-5 0.1 10 ${i}

echo "./cpp-data/small-gsvm-0"
./estimator ./cpp-data/small-gsvm-0 0.1 10 ${i}
echo "./cpp-data/small-gsvm-1"
./estimator ./cpp-data/small-gsvm-1 0.1 10 ${i}
echo "./cpp-data/small-gsvm-2"
./estimator ./cpp-data/small-gsvm-2 0.1 10 ${i}
echo "./cpp-data/small-gsvm-3"
./estimator ./cpp-data/small-gsvm-3 0.1 10 ${i}
echo "./cpp-data/small-gsvm-4"
./estimator ./cpp-data/small-gsvm-4 0.1 10 ${i}
echo "./cpp-data/small-gsvm-5"
./estimator ./cpp-data/small-gsvm-5 0.1 10 ${i}
echo "./cpp-data/small-gsvm-6"
./estimator ./cpp-data/small-gsvm-6 0.1 10 ${i}

echo "./cpp-data/large-mrvm-0"
./estimator ./cpp-data/large-mrvm-0 5e5 100 ${i}
echo "./cpp-data/large-mrvm-1"
./estimator ./cpp-data/large-mrvm-1 5e5 100 ${i}
echo "./cpp-data/large-mrvm-2"
./estimator ./cpp-data/large-mrvm-2 5e5 100 ${i}
echo "./cpp-data/large-mrvm-3"
./estimator ./cpp-data/large-mrvm-3 1e6 100 ${i}
echo "./cpp-data/large-mrvm-4"
./estimator ./cpp-data/large-mrvm-4 1e6 100 ${i}
echo "./cpp-data/large-mrvm-5"
./estimator ./cpp-data/large-mrvm-5 1e6 100 ${i}
echo "./cpp-data/large-mrvm-6"
./estimator ./cpp-data/large-mrvm-6 1e6 100 ${i}
echo "./cpp-data/large-mrvm-7"
./estimator ./cpp-data/large-mrvm-7 1e7 100 ${i}
echo "./cpp-data/large-mrvm-8"
./estimator ./cpp-data/large-mrvm-8 1e7 100 ${i}
echo "./cpp-data/large-mrvm-9"
./estimator ./cpp-data/large-mrvm-9 1e7 100 ${i}

done
