# For HW1

Python Environment:

numpy==1.21.6

opencv-python==4.6.0.66

Python == 3.10

For DoG.py, you can run following code to use eval.py to evaluate your DoG.py

python3 eval.py --threshold 3.0 --image_path ./testdata/1.png --gt_path ./testdata/1_gt.npy

For DoG.py, you can run following code to use eval.py to evaluate your JBF.py

python3 eval.py --image_path ./testdata/ex.png --gt_bf_path ./testdata/ex_gt_bf.png --gt_jbf_path ./testdata/ex_gt_bf.png

# For HW2:

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
