sudo apt update
sudo apt install python3
sudo apt install python3-pip
sudo apt install python3-virtualenv
virtualenv rein_motif_algorithm_env
source rein_motif_algorithm_env/bin/activate
pip3 install argparse numpy pandas networkx matplotlib
cd gtrieScannerFolder
make
cd ..
