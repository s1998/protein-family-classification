echo "Assuming the data is saved as uniprot-all.tab and pwd is main folder"

cd data
echo "cd data"
git clone https://github.com/stanfordnlp/GloVe.git
echo "git clone https://github.com/stanfordnlp/GloVe.git"
cd GloVe/
echo "cd GloVe/"
make
echo "make"
cd ../../utils
echo "cd ../../utils"
python3 script1.py
echo "python3 script1.py"
cd ./../data
echo "cd ./../data"
./GloVe/build/vocab_count -min-count 1 -verbose 2 < all_seq_in_one.txt > vocab
echo "./GloVe/build/vocab_count -min-count 1 -verbose 2 < all_seq_in_one.txt > vocab"
./GloVe/build/cooccur -memory 2 -verbose 2 < vocab > coocur
echo "./GloVe/build/cooccur -memory 2 -verbose 2 < vocab > coocur"
./GloVe/build/shuffle -memory 2 -verbose 2 <coocur> coocur_shuffled
echo "./GloVe/build/shuffle -memory 2 -verbose 2 <coocur> coocur_shuffled"
./GloVe/build/glove -save-file vectors_pfam.txt -threads 4 -input-file coocur_shuffled -x-max 20 -iter 100 -vector-size 100 -binary 2 -vocab-file vocab -verbose 2
echo "./GloVe/build/glove -save-file vectors_pfam.txt -threads 4 -input-file coocur_shuffled -x-max 20 -iter 100 -vector-size 100 -binary 2 -vocab-file vocab -verbose 2"

echo "Embeddings for all trigrams created using glove"


cd ./../utils
echo "cd ./../utils"
python3 script2.py
echo "python3 script2.py"

echo "Preprocessing complete"

python3 model.py
