#python run_inference.py \
#	-i /media/DATA/VAD_datasets/taiwan_sa/testing/frames \
#	-o /media/DATA/VAD_datasets/taiwan_sa/testing/mask_rcnn_detections \
#	--for_deepsort \
#	--image_shape 1280 720 3 \
#	-g=0 
python run_inference.py \
	-i /media/DATA/AnAnAccident_Detection_Dataset/frames \
	-o /media/DATA/AnAnAccident_Detection_Dataset/mask_rcnn_detections \
	--for_deepsort \
	--image_shape 1280 720 3 \
	-g=0 
