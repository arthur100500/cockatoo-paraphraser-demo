"""Module for a Parrot Model class"""

import torch

from parrot import Parrot


class ParrotModel:
    """A class for the parrot model"""

    def __init__(self, seed: int = 0) -> None:
        """Creates parrot"""
        self.parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)
        self.seed = seed
        self.amount = 1
        self.__set_seed()
        self.fluency_threshold = 0.90
        self.adequacy_threshold = 0.99
        self.do_diverse = False

    def set_seed(self, seed: int) -> None:
        """Sets seed for parrot and updates it in cuda and torch"""
        self.seed = seed
        self.__set_seed()

    def __set_seed(self) -> None:
        """Sets seed for torch"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def para_phrase(self, phrase: str) -> list:
        """Returns list of paraphrases"""
        para_phrases = self.parrot.augment(
            input_phrase=phrase,
            diversity_ranker="levenshtein",
            do_diverse=self.do_diverse,
            max_return_phrases=self.amount,
            fluency_threshold=self.fluency_threshold,
            adequacy_threshold=self.adequacy_threshold
        )

        return para_phrases
