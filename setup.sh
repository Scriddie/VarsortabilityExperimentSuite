# create virtual environment
sudo rm -rf env
python3 -m venv env
# install requirements
source env/bin/activate
easy_install -U "pip==21.3" 
pip install -r requirements.txt
# install dependencies
cd src
git clone https://github.com/xunzheng/notears
git clone https://github.com/ignavierng/golem