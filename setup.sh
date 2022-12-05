# check if the venv is created
if [ ! -d "venv" ]; then
    # create the venv
    python3 -m venv venv
fi

# activate the venv
. venv/bin/activate

# install the requirements
pip install -r requirements.txt

# check if exists the folder test and train
if [ ! -d "data/train" ]; then
   # unzip the dataset in the folder data
    unzip data
fi

