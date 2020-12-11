# Chid_Bert_baseline
A based-bert baseline for Chinese idiom cloze test with pytorch.
## Chinese Idiom Reading Comprehension Competition  
[the competition official website](https://www.biendata.net/competition/idiom/)

## paper
The ChID Dataset for paper [ChID: A Large-scale Chinese IDiom Dataset for Cloze Test.](https://www.aclweb.org/anthology/P19-1075/)

## official baseline and paper code
[ChID-Dataset](https://github.com/chujiezheng/ChID-Dataset)

## this baseline code
use transformers and pytorch implement based-bert for chinese idiom cloze test

## requirements
pyhton3.6  
torch=1.1.0  
transformers==2.8.0  
scikit-learn==0.22.2.post1  
pandas==1.0.3  
tqdm==4.45.0  

## dataset Download
[Chid dataset download](https://drive.google.com/drive/folders/1qdcMgCuK9d93vLVYJRvaSLunHUsGf50u )  
save chid data into ./data 
you maybe need a vpn

## pretrain model
[download]( https://huggingface.co/models)  
For this baseline, we use chinese_wwm_pytorch as pretrain model
save chid data into ./pretrained_models

