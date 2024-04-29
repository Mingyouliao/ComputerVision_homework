

# For Hw2:

conda create -n cv2024_hw2 python==3.6.13
conda activate cv2024_hw2
conda install menpo::cyvlfeat
pip install -r requirements.txt

### For p1, you can run the following code:
python3 p1.py --feature tiny_image  --classifier nearest_neighbor --dataset_dir ../hw2_data/p1_data/
python3 p1.py --feature bag_of_sift --classifier nearest_neighbor --dataset_dir ../hw2_data/p1_data/

### For p2, you can run the following code:
Training (only you will run this)
python3 p2_train.py --dataset_dir ../hw2_data/p2_data/

Manually put your best model under folder checkpoint/ and rename as 'resnet18_best.pth' or 'mynet_best.pth'
Inference
python3 p2_inference.py --test_datadir ../hw2_data/p2_data/val --model_type resnet18 --output_path ./output/pred.csv
python3 p2_inference.py --test_datadir ../hw2_data/p2_data/val --model_type mynet --output_path ./output/pred.csv

Evaluation (TA will also run this)
python3 p2_eval.py --csv_path ./output/pred.csv --annos_path ../hw2_data/p2_data/val/annotations.json
