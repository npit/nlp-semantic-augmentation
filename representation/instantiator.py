from representation.bag_representation import (BagRepresentation,
                                               TFIDFRepresentation)
from representation.document_embedding import DocumentEmbedding
from representation.existing_vectors import ExistingVectors
from representation.word_embedding import WordEmbedding


class Instantiator:
    component_name = "representation"

    def create(config):
        name = config.name
        if name == BagRepresentation.name:
            return BagRepresentation(config)
        if name == TFIDFRepresentation.name:
            return TFIDFRepresentation(config)
        if name == DocumentEmbedding.name:
            return DocumentEmbedding(config)
        if name == ExistingVectors.name:
            return ExistingVectors(config)

        # any unknown name, if it's an absolute path it's path to existing vectors
        # if isabs(name
        # else, is assumed to be an embedding name, i.e. pretrained word embeddings
        return WordEmbedding(config)
