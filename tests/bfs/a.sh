mkdir -p eval/error_dump
./bin/test_bfs_7.5_x86_64 market /data/gunrock_dataset/large/soc-LiveJournal1/soc-LiveJournal1.mtx --undirected --src=largestdegree --queue-sizing=0 --in-sizing=0 --traversal-mode=1 --device=0,1 --partition-seed=1444776384
