KAGGLE_API:=''

.PHONY: install-dependences
install-dependences: install-dependences
	pip3 install -r requirements.txt

.PHONY: install-data
install-data: install-data
	mkdir -p ~/.kaggle
	echo $(KAGGLE_API) > ~/.kaggle/kaggle.json
	chmod 600 ~/.kaggle/kaggle.json
	kaggle datasets download -d jessicali9530/celeba-dataset
	kaggle datasets download -d vladislavrr/Celeba-Masks
	mkdir data
	unzip celeba-dataset.zip -d data
	unzip Celeba-Masks.zip -d data
	rm -rf celeba-dataset.zip Celeba-Masks.zip
