download-datasets:
	gdown https://drive.google.com/drive/u/3/folders/1rGblqA0Wh0vhDFrjasMGvOYpyDiD3jVW -O ./ --folder 

olid-default:
	python train.py olidv1 \
		--min_confidence_threshold 0.9 \
		--exp_name olidv1-default

olid-ns:
	python train.py olidv1 \
		--min_confidence_threshold 0.9 \
		--use_augmentation \
		--increase_attention_dropout_amount 0.15 \
		--increase_classifier_dropout_amount 0.15 \
		--exp_name olidv1-ns

convabuse-default:
	python train.py convabuse \
		--min_confidence_threshold 0.8 \
		--exp_name convabuse-default

convabuse-ns:
	python train.py convabuse \
		--min_confidence_threshold 0.8 \
		--use_augmentation \
		--increase_attention_dropout_amount 0.15 \
		--increase_classifier_dropout_amount 0.15 \
		--exp_name convabuse-ns

mhs-default:
	python train.py mhs \
		--min_confidence_threshold 0.8 \
		--exp_name mhs-default

mhs-ns:
	python train.py mhs \
		--min_confidence_threshold 0.8 \
		--use_augmentation \
		--increase_attention_dropout_amount 0.15 \
		--increase_classifier_dropout_amount 0.15 \
		--exp_name mhs-ns