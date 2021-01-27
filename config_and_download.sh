echo "Creating directories"
mkdir -p DATA/DBLP
mkdir -p DATA/wikipedia
mkdir -p DATA/blogCatalog
mkdir -p DATA/LFR1
mkdir -p DATA/karate
mkdir -p DATA/football



echo "Download LFR dataset"
wget -q http://webia.lip6.fr/~gerald/data/rcome/LFR1/edges.txt -O  ./DATA/LFR1/edges.txt
wget -q http://webia.lip6.fr/~gerald/data/rcome/LFR1/communities.txt -O ./DATA/LFR1/communities.txt

echo "Download karate dataset"
wget -q http://webia.lip6.fr/~gerald/data/rcome/karate/edges.txt -O ./DATA/karate/edges.txt
wget -q http://webia.lip6.fr/~gerald/data/rcome/karate/communities.txt -O ./DATA/karate/communities.txt


echo "Download football dataset"
wget -q http://webia.lip6.fr/~gerald/data/rcome/football/edges.txt -O ./DATA/football/edges.txt
wget -q http://webia.lip6.fr/~gerald/data/rcome/football/communities.txt -O ./DATA/football/communities.txt


echo "Download DBLP dataset"
wget -q https://github.com/vwz/ComE/blob/master/data/Dblp/Dblp.mat?raw=true -O ./DATA/DBLP/Dblp.mat
wget -q https://raw.githubusercontent.com/vwz/ComE/master/data/Dblp/Dblp.labels -O ./DATA/DBLP/labels.txt

echo "Download wikipedia dataset"
wget -q http://snap.stanford.edu/node2vec/POS.mat -O ./DATA/wikipedia/wikipedia.mat

echo "Download blog-catalog dataset..."
wget -q https://github.com/quark0/TAE/raw/master/data/BlogCatalog-dataset/data/edges.csv -O ./DATA/blogCatalog/edges.csv
wget -q https://github.com/quark0/TAE/raw/master/data/BlogCatalog-dataset/data/group-edges.csv -O ./DATA/blogCatalog/group-edges.csv
wget -q https://github.com/quark0/TAE/raw/master/data/BlogCatalog-dataset/data/groups.csv -O ./DATA/blogCatalog/groups.csv
wget -q https://github.com/quark0/TAE/raw/master/data/BlogCatalog-dataset/data/nodes.csv -O ./DATA/blogCatalog/nodes.csv
