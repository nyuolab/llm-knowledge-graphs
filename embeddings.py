# NYU OLAB, 2023

# imports
import torch
from transformers import AutoModel, AutoTokenizer
from chromadb import Documents, Embeddings, EmbeddingFunction
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class HuggingFaceEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str, max_length: int = 64, device: str = None):
        """
        Base class for chromadb embedding functions that use HuggingFace models

        :param model_name: the name of the HuggingFace model to use
        :param max_length: the maximum length of the input text (default: 64)
        :param device: the device to use (default: None --> behavior: cpu, but use gpu if available)
        """
        super().__init__()

        # set model and tokenizer
        self._setup_model_tokenizer(model_name=model_name)

        # set device
        if device is not None:
            self.device = device
        else:
            # default to cpu, but use gpu if available
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"

        # move model to device
        self.torch_model.to(self.device)

        # set max length
        self.max_length = max_length

    def __call__(self, input: Documents) -> Embeddings:
        # return embeddings
        return self.get_embeddings(input)

    def _setup_model_tokenizer(self, model_name: str) -> None:
        """
        Sets up the model and tokenizer for the embedding function
        """
        self.torch_model = AutoModel.from_pretrained(model_name)
        self.torch_tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_embeddings(self, queries) -> list:
        """
        Consumes a list of queries
        Produces a list of embeddings

        :param queries: a list of queries
        :return: a list of embeddings
        """

        with torch.no_grad():
            # tokenize the queries
            encoded = self.torch_tokenizer(
                queries,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)

            # encode the queries (use the [CLS] last hidden states as the representations)
            # move to cpu
            embeds = self.torch_model(**encoded).last_hidden_state[:, 0, :].to("cpu")

            # normalize
            # embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)

            # convert from torch tensor to list
            embeds = embeds.tolist()

            return embeds


class MedCPTEmbeddding(HuggingFaceEmbedding):
    """
    Class for the MedCPT-Query-Encoder embedding function for ChromaDB

    MedCPT Model Doc :: https://huggingface.co/ncbi/MedCPT-Query-Encoder
    """

    def __init__(self, **kwargs):
        super().__init__(model_name="ncbi/MedCPT-Query-Encoder", **kwargs)


class BertBaseEmbedding(HuggingFaceEmbedding):
    """
    Class for the BertBase embedding function for ChromaDB

    BertBase Model Doc :: https://huggingface.co/bert-base-uncased
    """

    def __init__(self, **kwargs):
        super().__init__(model_name="bert-base-uncased", **kwargs)


class ClinicalBertEmbedding(HuggingFaceEmbedding):
    """
    Class for the ClinicalBert embedding function for ChromaDB

    ClinicalBert Model Doc :: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
    """

    def __init__(self, **kwargs):
        super().__init__(model_name="emilyalsentzer/Bio_ClinicalBERT", **kwargs)


EMBED_MODEL_DICT = {
    "medcpt": MedCPTEmbeddding(),
    "clinicalbert": ClinicalBertEmbedding(),
    "bertbase": BertBaseEmbedding(),
    "default": DefaultEmbeddingFunction(),
}
