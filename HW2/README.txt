Environment Setup (conda is strongly suggested)

For Windows/Mac/Linux

> conda create -n cv2024_hw2 python==3.6.13
> conda activate cv2024_hw2
> conda install menpo::cyvlfeat
> pip install -r requirements.txt

### example for running the python files, you don't have to submit this file
python3 p1.py --feature tiny_image  --classifier nearest_neighbor --dataset_dir ../hw2_data/p1_data/
python3 p1.py --feature bag_of_sift --classifier nearest_neighbor --dataset_dir ../hw2_data/p1_data/
