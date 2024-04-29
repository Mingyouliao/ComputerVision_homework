Python Environment:
numpy==1.21.6
opencv-python==4.6.0.66
Python == 3.10

For DoG.py, you can run following code to use eval.py to evaluate your DoG.py
python3 eval.py --threshold 3.0 --image_path ./testdata/1.png --gt_path ./testdata/1_gt.npy

For DoG.py, you can run following code to use eval.py to evaluate your JBF.py
python3 eval.py --image_path ./testdata/ex.png --gt_bf_path ./testdata/ex_gt_bf.png --gt_jbf_path ./testdata/ex_gt_bf.png



