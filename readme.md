

Label 0:
select motif: 0.999
python sample.py --nsample 1000 --vocab vocab_0.txt --mask checkpoints/mask_0.pt --graph_label 0 --motif_embedding checkpoints/motif_embedding_0.pt --hidden 64 --model vae_model/Mutagenicity_0/05/model.iter-500 > mol_samples_0_14.txt

python sample.py --nsample 1000 --vocab vocab_0.txt --mask checkpoints/mask_0.pt --graph_label 0 --motif_embedding checkpoints/motif_embedding_0.pt --hidden 64 --model vae_model/Mutagenicity_0/05/model.iter-8500 > mol_samples_0_21.txt


Label 1:
select motif: 0.999999
python sample.py --nsample 10000 --vocab vocab_1.txt --mask checkpoints/mask_1.pt --graph_label 1 --motif_embedding checkpoints/motif_embed
ding_1.pt --hidden 64 --model vae_model/Mutagenicity_1/08/model.iter-6000 > mol_samples_1_11.txt

select motif: 0.999
python sample.py --nsample 10000 --vocab vocab_1.txt --mask checkpoints/mask_1.pt --graph_label 1 --motif_embedding checkpoints/motif_embedding_1.pt --hidden 64 --model vae_model/Mutagenicity_1/08/model.iter-8000 > mol_samples_1_13.txt

python sample.py --nsample 10000 --vocab vocab_1.txt --mask checkpoints/mask_1.pt --graph_label 1 --motif_embedding checkpoints/motif_embedding_1.pt --hidden 64 --model vae_model/Mutagenicity_1/08/model.iter-7000 > mol_samples_1_17.txt

select motif: 0.99
python sample.py --nsample 10000 --vocab vocab_1.txt --mask checkpoints/mask_1.pt --graph_label 1 --motif_embedding checkpoints/motif_embedding_1.pt --hidden 64 --model vae_model/Mutagenicity_1/08/model.iter-8000 > mol_samples_1_19.txt
