
from motifpiece import MotifPiece
from torch_geometric.data import Data

class MomoX:
    def __init__(self, dataset, target_model) -> None:
        self.dataset = dataset
        self.model = target_model
        self.process()

    
    def process(self):
        motif_piece = MotifPiece(self.dataset)
        motif_list = motif_piece.motifs_list
        num_motifs = len(motif_piece.motif_explanation)
        print(num_motifs)